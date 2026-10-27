"""Run a local frozen manifest with the pinned Prism CPU backend and deadline.

No downloads, model changes or alternative scoring algorithms. An explicit
SHA-bound, data-free profile selects existing-data resource probes; unsuitable
profiles are rejected before model loading. POSIX process-group cleanup only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_SHA = 'c9c34bc9c4042c13a31598ebf43461cd0ad8f00d'
PROFILE_FORMAT = 'openbonsai.prism-cpu-profile.v1'
MAX_SECONDS = 10800.0
FACTOR = 1.25
RESERVE = 300.0
SOURCE_FILES = ('openbonsai/evaluation.py', 'openbonsai/eval_backends.py',
                'tools/prism/prism_eval_worker.cpp')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + '\n').encode('utf-8')


def write_new(path, value):
    with Path(path).open('xb') as handle:
        handle.write(encode(value))


def update_owned(path, value):
    # Only receipts created in this exclusively new output directory are updated.
    temporary = Path(str(path) + '.tmp')
    with temporary.open('xb') as handle:
        handle.write(encode(value))
    os.replace(temporary, path)


def jsonable(value):
    if isinstance(value, bytes):
        return {'bytes_hex': value.hex()}
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def check_deadline(deadline, clock=time.monotonic):
    remaining = deadline - clock()
    if not math.isfinite(remaining) or remaining <= 0:
        raise TimeoutError('Whole-process monotonic deadline exhausted')
    return remaining


def sha_string(value):
    require(isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value), 'Expected lowercase SHA256')
    return value


def exact_keys(value, names, label):
    require(isinstance(value, dict) and set(value) == set(names), 'Invalid ' + label + ' fields')


def integer(value, minimum, maximum, label):
    require(type(value) is int and minimum <= value <= maximum, 'Invalid ' + label)
    return value


def score_shape(item):
    positions = item['score_positions']
    batches = {}
    for target in positions:
        batch = (target - 1) // 64
        batches[batch] = batches.get(batch, 0) + 1
    return {'input_tokens': len(item['input_ids']), 'decoder_positions': positions[-1],
            'full_vocab_rows': len(positions), 'max_rows_per_batch': max(batches.values())}


def dominates(probe, other):
    return all(probe[key] >= other[key] for key in probe)


def build_plan(manifest, profile, *, evaluation=None):
    """Validate explicit probe identities and derive every workload count.

    A longest token sequence alone is insufficient: score-row count and per-batch
    output density also matter. No automatic probe substitution is permitted.
    """
    if evaluation is None:
        from openbonsai import evaluation
    exact_keys(profile, {'format', 'manifest_sha256', 'context_length', 'vocab_size',
                         'threads', 'resource_probes'}, 'profile')
    require(profile['format'] == PROFILE_FORMAT, 'Unsupported profile format')
    require(sha_string(profile['manifest_sha256']) == manifest.get('frozen_sha256'),
            'Profile belongs to a different frozen manifest')
    context = integer(profile['context_length'], 2, 1048576, 'context_length')
    vocab = integer(profile['vocab_size'], 2, 1048576, 'vocab_size')
    integer(profile['threads'], 1, 4, 'CPU threads (supported range 1..4)')
    evaluation.validate_manifest(manifest, context_length=context, vocab_size=vocab)
    require(manifest['role'] in ('search_validation', 'evaluation'),
            'Calibration-role inputs are not an evaluation manifest')
    require(manifest['nll'] and manifest['choices'] and manifest['generations'],
            'This bounded profile requires NLL, choices and generation records')
    probes = profile['resource_probes']
    exact_keys(probes, {'nll', 'choice', 'generation'}, 'resource probes')
    exact_keys(probes['nll'], {'record_index'}, 'NLL probe')
    nll_index = integer(probes['nll']['record_index'], 0, len(manifest['nll']) - 1, 'NLL index')
    nll = manifest['nll'][nll_index]
    exact_keys(probes['choice'], {'question_index', 'option_index'}, 'choice probe')
    question_index = integer(probes['choice']['question_index'], 0, len(manifest['choices']) - 1, 'question index')
    question = manifest['choices'][question_index]
    option_index = integer(probes['choice']['option_index'], 0, len(question['options']) - 1, 'option index')
    choice = question['options'][option_index]
    options = [option for row in manifest['choices'] for option in row['options']]
    nll_shape, choice_shape = score_shape(nll), score_shape(choice)
    require(all(dominates(nll_shape, score_shape(row)) for row in manifest['nll']),
            'NLL probe does not dominate every input/decoder/logit-row/batch shape')
    require(all(dominates(choice_shape, score_shape(row)) for row in options),
            'Choice probe does not dominate every input/decoder/logit-row/batch shape')
    generation = probes['generation']
    exact_keys(generation, {'text_index', 'token_start', 'token_end', 'target_position'}, 'generation probe')
    text_index = integer(generation['text_index'], 0, len(manifest['texts']) - 1, 'generation source text index')
    source = manifest['texts'][text_index]
    start = integer(generation['token_start'], 0, len(source['input_ids']) - 2, 'generation slice start')
    end = integer(generation['token_end'], start + 2, len(source['input_ids']), 'generation slice end')
    ids = source['input_ids'][start:end]
    target = integer(generation['target_position'], 1, len(ids) - 1, 'generation target')
    require(target == len(ids) - 1, 'Generation resource probe must read the last source logit')
    evaluation.validate_score_positions(ids, [target], context_length=context, vocab_size=vocab)
    max_forward = max(len(row['input_ids']) + row['max_new_tokens'] - 1 for row in manifest['generations'])
    max_guard = max(len(row['input_ids']) + row['max_new_tokens'] for row in manifest['generations'])
    require(target >= max_forward, 'Generation resource probe is shorter than a capped prefix')
    # The source must be part of the existing scored/generated inputs, not an
    # extra unscored text injected merely as a synthetic probe reservoir.
    referenced = {row['text_id'] for row in manifest['nll'] + manifest['generations']}
    referenced.update(row['text_id'] for row in options)
    require(source['id'] in referenced, 'Generation probe source is not an existing evaluation input')
    workload = {'nll_records': len(manifest['nll']),
                'nll_targets': sum(len(row['score_positions']) for row in manifest['nll']),
                'choice_questions': len(manifest['choices']),
                'choice_labelled': sum('answer_id' in row for row in manifest['choices']),
                'choice_option_calls': len(options),
                'choice_targets': sum(len(row['score_positions']) for row in options),
                'generation_records': len(manifest['generations']),
                'generation_tokens_maximum': sum(row['max_new_tokens'] for row in manifest['generations']),
                'generation_forward_positions_maximum': max_forward,
                'generation_context_guard_maximum': max_guard,
                'tokenization_texts': len(manifest['texts'])}
    resource = [
        {'name': 'nll', 'input_ids': nll['input_ids'], 'score_positions': nll['score_positions'],
         'identity': {'record_index': nll_index, 'record_id': nll['id']}, 'shape': nll_shape},
        {'name': 'choice', 'input_ids': choice['input_ids'], 'score_positions': choice['score_positions'],
         'identity': {'question_index': question_index, 'option_index': option_index,
                      'question_id': question['id'], 'option_id': choice['id']}, 'shape': choice_shape},
        {'name': 'generation_last_logit', 'input_ids': ids, 'score_positions': [target],
         'identity': {**generation, 'source_text_id': source['id']},
         'shape': score_shape({'input_ids': ids, 'score_positions': [target]})}]
    return {'workload': workload, 'probes': resource, 'context_length': context,
            'vocab_size': vocab, 'threads': profile['threads'],
            'probe_scope': 'resource-only existing-token inputs; excluded from final quality aggregation',
            'forecast_limit': 'shape dominance plus margin is an estimate, not a runtime guarantee'}


def forecast(plan, seconds, remaining):
    for key in ('nll', 'choice', 'generation_last_logit', 'tokenization'):
        value = seconds.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                'Invalid measured duration ' + key)
    require(math.isfinite(remaining) and remaining > 0, 'Invalid remaining budget')
    work = plan['workload']
    components = {'nll': work['nll_records'] * seconds['nll'],
                  'choice_independent_options': work['choice_option_calls'] * seconds['choice'],
                  'generation_full_cap': work['generation_tokens_maximum'] * seconds['generation_last_logit'],
                  'repeated_full_tokenization': seconds['tokenization']}
    total = FACTOR * math.fsum(components.values()) + RESERVE
    return {'components_seconds': components, 'measured_probe_seconds': seconds,
            'factor': FACTOR, 'reserve_seconds': RESERVE, 'projected_remaining_seconds': total,
            'remaining_seconds_at_admission': remaining, 'admitted': total <= remaining,
            'early_eos_assumed': False, 'workload': work, 'forecast_is_not_a_guarantee': True}


class RecordedBackend:
    def __init__(self, backend, output, deadline, manifest_sha, clock=time.monotonic):
        self.backend, self.output, self.deadline = backend, Path(output), deadline
        self.manifest_sha, self.clock = manifest_sha, clock
        self.output.mkdir(exist_ok=False)
        self.phase, self.calls, self.last_receipt = 'tokenization-gate', 0, None

    def call(self, operation, request, function):
        remaining = check_deadline(self.deadline, self.clock)
        self.calls += 1
        path = self.output / f'{self.calls:04d}-{operation}.json'
        receipt = {'index': self.calls, 'phase': self.phase, 'operation': operation,
                   'request': copy.deepcopy(request), 'status': 'started',
                   'manifest_sha256': self.manifest_sha, 'remaining_before_seconds': remaining}
        write_new(path, receipt)
        started = self.clock()
        try:
            result = function()
            check_deadline(self.deadline, self.clock)
            receipt.update(status='complete', result=jsonable(result))
            return result
        except BaseException as error:
            receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            receipt['seconds'] = self.clock() - started
            receipt['remaining_after_seconds'] = max(0., self.deadline - self.clock())
            update_owned(path, receipt)
            self.last_receipt = receipt

    def metadata(self):
        return self.call('metadata', {}, self.backend.metadata)

    def tokenize(self, text, parse_special=False):
        return self.call('tokenize', {'text': text, 'parse_special': parse_special, 'add_special': False},
                         lambda: self.backend.tokenize(text, parse_special))

    def token_pieces(self, ids):
        return self.call('token_pieces', {'ids': ids}, lambda: self.backend.token_pieces(ids))

    def score_tokens(self, ids, positions):
        return self.call('score', {'input_ids': ids, 'score_positions': positions},
                         lambda: self.backend.score_tokens(ids, positions))

    def generate(self, ids, maximum, eos):
        return self.call('generate', {'input_ids': ids, 'max_new_tokens': maximum, 'eos_ids': eos},
                         lambda: self.backend.generate(ids, maximum, eos))


def evaluate_plan(manifest, profile, backend, output, deadline, *, clock=time.monotonic, evaluation=None):
    if evaluation is None:
        from openbonsai import evaluation
    plan = build_plan(manifest, profile, evaluation=evaluation)
    output = Path(output)
    write_new(output / 'derived-plan.json', plan)
    wrapped = RecordedBackend(backend, output / 'calls', deadline, manifest['frozen_sha256'], clock)
    started = clock()
    verified = evaluation.verify_tokenization(wrapped, manifest)
    tokenization_seconds = clock() - started
    require(verified['texts'] == plan['workload']['tokenization_texts']
            and verified['exact_ids'] and verified['exact_raw_pieces'], 'Incomplete native tokenization gate')
    write_new(output / 'native-tokenization-gate.json', dict(status='passed', seconds=tokenization_seconds, **verified))
    wrapped.phase = 'resource-preflight-not-quality'
    seconds, measurements = {'tokenization': tokenization_seconds}, []
    for probe in plan['probes']:
        result = wrapped.score_tokens(probe['input_ids'], probe['score_positions'])
        evaluation.validate_score_result(result, probe['input_ids'], probe['score_positions'])
        measured = wrapped.last_receipt['seconds']
        seconds[probe['name']] = measured
        measurements.append({**probe, 'seconds': measured, 'call_index': wrapped.calls,
                             'purpose': 'resource-only; excluded from final quality aggregation'})
    write_new(output / 'resource-preflight.json', {'status': 'complete', 'probes': measurements})
    admission = forecast(plan, seconds, check_deadline(deadline, clock))
    write_new(output / 'admission.json', admission)
    if not admission['admitted']:
        raise RuntimeError('Full fixed manifest does not fit the remaining budget; no caps/data changed')
    wrapped.phase = 'full-fixed-evaluation-suite'
    report = evaluation.evaluate_manifest(wrapped, manifest)
    work = plan['workload']
    require(report['aggregate']['scored_tokens'] == work['nll_targets']
            and report['aggregate']['choice_labelled'] == work['choice_labelled']
            and report['aggregate']['generations'] == work['generation_records']
            and len(report['nll']) == work['nll_records'] and len(report['choices']) == work['choice_questions'],
            'Incomplete final suite')
    write_new(output / 'evaluation.json', report)
    return {'status': 'complete', 'calls': wrapped.calls, 'aggregate': report['aggregate'],
            'admission': admission, 'role': manifest['role'],
            'legacy_fp32_cross_entropy_compatible': False,
            'scope': 'provided frozen inputs plus pinned CPU backend; no model-equivalence claim'}


def read_pinned_json(path, expected_sha):
    data = Path(path).read_bytes()
    require(hashlib.sha256(data).hexdigest() == sha_string(expected_sha), 'File SHA mismatch: ' + str(path))
    return json.loads(data)


def check_local_inputs(args):
    manifest = read_pinned_json(args.manifest, args.expected_manifest_sha256)
    profile = read_pinned_json(args.profile, args.expected_profile_sha256)
    require(digest(args.worker) == sha_string(args.expected_worker_sha256), 'Worker executable SHA mismatch')
    sha_string(args.expected_gguf_sha256)
    require(Path(args.gguf).is_file(), 'GGUF must already exist locally')
    git = lambda *argv: subprocess.check_output(
        ['git', '-C', str(args.prism_source), *argv], text=True, timeout=10).strip()
    require(git('rev-parse', 'HEAD') == RUNTIME_SHA and not git('status', '--porcelain'),
            'Prism source pin/cleanliness mismatch')
    sys.path.insert(0, str(ROOT))
    from openbonsai import evaluation
    plan = build_plan(manifest, profile, evaluation=evaluation)
    return manifest, profile, plan


def child_main(jobpath):
    job = json.loads(Path(jobpath).read_bytes())
    output, args = Path(job['output']), argparse.Namespace(**job['arguments'])
    require(os.getppid() == job['supervisor_pid'] and os.getpgrp() == os.getpid(),
            'Child must run under its isolated supervisor process group')
    require(digest(__file__) == job['runner_sha256'], 'Runner changed after source freeze')
    for name, expected in job['code_sha256'].items():
        require(digest(ROOT / name) == expected, 'Evaluator source changed: ' + name)
    deadline, backend = job['deadline_monotonic'], None
    report = {'status': 'starting', 'legacy_fp32_cross_entropy_compatible': False}
    write_new(output / 'child-status.json', report)
    try:
        check_deadline(deadline)
        manifest, profile, plan = check_local_inputs(args)
        import torch
        torch.set_num_threads(plan['threads'])
        from openbonsai.eval_backends import PrismBackend
        from openbonsai import evaluation
        startup, remaining = time.monotonic(), check_deadline(deadline)
        report.update(status='starting-worker', request_timeout_seconds=remaining)
        update_owned(output / 'child-status.json', report)
        backend = PrismBackend(args.worker, args.gguf, context_length=plan['context_length'],
            runtime_source_dir=args.prism_source, stderr_path=output / 'prism.stderr',
            threads=plan['threads'], gpu_layers=0, runtime_sha=RUNTIME_SHA,
            expected_executable_sha256=args.expected_worker_sha256,
            expected_model_sha256=args.expected_gguf_sha256, request_timeout=remaining)
        metadata = backend.metadata()
        settings = {'cuda_build': False, 'gpu_layers': 0, 'threads': plan['threads'],
                    'context_length': plan['context_length'], 'vocab_size': plan['vocab_size'],
                    'kv_cache_dtype': 'float32', 'flash_attention': False, 'load_mtp': False,
                    'batch_size': 64, 'ubatch_size': 64}
        for key, value in settings.items():
            require(type(metadata.get(key)) is type(value) and metadata[key] == value, 'Runtime setting mismatch ' + key)
        write_new(output / 'worker-startup.json', {'seconds': time.monotonic() - startup,
            'metadata': metadata, 'request_timeout_seconds': remaining,
            'effective_request_deadline': 'bounded by external whole-process supervisor',
            'startup_includes_full_gguf_and_executable_hashing': True, 'worker_pid': backend._process.pid})
        report.update(evaluate_plan(manifest, profile, backend, output, deadline, evaluation=evaluation))
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    finally:
        if backend is not None:
            started = time.monotonic()
            try:
                backend.close()
                report['backend_close_complete'] = True
            except BaseException as error:
                report.update(status='failed', backend_close_error=f'{type(error).__name__}: {error}')
            report['backend_close_seconds'] = time.monotonic() - started
        report['elapsed_from_supervisor_start_seconds'] = time.monotonic() - job['started_monotonic']
        report['deadline_remaining_seconds'] = max(0., deadline - time.monotonic())
        update_owned(output / 'child-status.json', report)
    return 0 if report['status'] == 'complete' else 1


def group_members(group):
    import psutil
    found = []
    # At cleanup the child may already have exited; enumerating PGID also catches
    # orphaned worker descendants that were reparented before the next sample.
    for proc in psutil.process_iter(['pid', 'create_time', 'status']):
        try:
            if os.getpgid(proc.pid) == group and proc.info['status'] != psutil.STATUS_ZOMBIE:
                found.append(proc)
        except (ProcessLookupError, PermissionError, psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return found


def cleanup_group(group, deadline, clock=time.monotonic):
    """TERM while time remains; KILL at deadline or after at most one second."""
    expired = clock() >= deadline
    immediate_kill = False
    if expired:
        # Enforce the execution boundary before potentially slower process enumeration.
        try:
            os.killpg(group, signal.SIGKILL)
            immediate_kill = True
        except ProcessLookupError:
            pass
    before = [p.pid for p in group_members(group)]
    if before and not expired:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
        end = min(deadline, clock() + 1.)
        while clock() < end and group_members(group):
            time.sleep(min(.05, max(0, end - clock())))
        if group_members(group):
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
    # Reaping/observation may occur just after the execution deadline; no model
    # process is granted extra computation for that administrative cleanup.
    until = time.monotonic() + 2.
    while group_members(group) and time.monotonic() < until:
        time.sleep(.05)
    after = [p.pid for p in group_members(group)]
    return {'group': group, 'pids_before_cleanup': before,
            'immediate_deadline_sigkill': immediate_kill,
            'live_pids_after_cleanup': after, 'no_live_group_members': not after}


def supervise_command(command, output, deadline, *, env=None, sample_interval=.5):
    """External process-group supervisor, also exercised with harmless fake processes."""
    import psutil
    output = Path(output)
    started = time.monotonic()
    reason = None
    process = None
    peak = 0
    supervisor_peak = 0
    known = {}
    samples = 0
    memory_path = output / 'rss-samples.jsonl'
    report_path = output / 'supervisor-process.json'
    initial = {
        'status': 'starting', 'command': command, 'deadline_monotonic': deadline,
        'process_group_isolated': True,
        'memory_method': 'sampled sum of runner+descendant RSS; shared pages can count more than once; not exact peak physical memory',
        'sample_interval_seconds': sample_interval, 'psutil_version': psutil.__version__,
    }
    write_new(report_path, initial)
    previous_handlers = {}

    def interrupted(signum, _frame):
        raise InterruptedError(f'Supervisor received signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGHUP):
        previous_handlers[sig] = signal.signal(sig, interrupted)
    try:
        check_deadline(deadline)
        with ((output / 'runner.stdout').open('xb') as stdout,
              (output / 'runner.stderr').open('xb') as stderr,
              memory_path.open('x') as memory):
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                       env=env, start_new_session=True)
            initial.update(status='running', pid=process.pid)
            update_owned(report_path, initial)
            while True:
                now = time.monotonic()
                if now >= deadline:
                    reason = 'whole-process-deadline'
                    break
                # Poll group membership rather than only the parent's descendants,
                # retaining visibility if a failed runner leaves an orphan worker.
                members = group_members(process.pid)
                rss = 0
                rows = []
                for proc in members:
                    try:
                        value = proc.memory_info().rss
                        rss += value
                        born = proc.create_time()
                        key = f'{proc.pid}:{born}'
                        known[key] = {
                            'pid': proc.pid, 'create_time': born,
                            'max_sampled_rss_bytes': max(
                                known.get(key, {}).get('max_sampled_rss_bytes', 0), value),
                        }
                        rows.append({'pid': proc.pid, 'rss_bytes': value})
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                peak = max(peak, rss)
                supervisor_peak = max(supervisor_peak, psutil.Process().memory_info().rss)
                samples += 1
                memory.write(json.dumps({'elapsed_seconds': now - started,
                                         'summed_rss_bytes': rss, 'processes': rows}) + '\n')
                memory.flush()
                if stdout.tell() > 4 << 20 or stderr.tell() > 4 << 20:
                    reason = 'runner-log-limit'
                    break
                if process.poll() is not None:
                    reason = 'exited'
                    break
                time.sleep(min(sample_interval, max(0., deadline - time.monotonic())))
    except BaseException as error:
        reason = f'supervisor-error:{type(error).__name__}:{error}'
    finally:
        cleanup = cleanup_group(process.pid, deadline) if process is not None else {
            'no_live_group_members': True, 'live_pids_after_cleanup': []}
        if process is not None:
            try:
                process.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.)
        result = {
            **initial,
            'status': 'complete' if reason == 'exited' and process.returncode == 0
                      and cleanup['no_live_group_members'] else 'failed',
            'reason': reason, 'exit_code': process.returncode if process else None,
            'cleanup': cleanup, 'elapsed_seconds': time.monotonic() - started,
            'sampled_peak_summed_rss_bytes': peak,
            'supervisor_sampled_peak_rss_bytes': supervisor_peak,
            'samples': samples, 'observed_processes': list(known.values()),
        }
        update_owned(report_path, result)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    return result


def supervisor_main(args):
    require(os.name == 'posix', 'POSIX process-group cleanup is required')
    require(type(args.max_seconds) in (int, float) and math.isfinite(args.max_seconds)
            and 0 < args.max_seconds <= MAX_SECONDS, 'max-seconds must be finite and in (0, 10800]')
    # Fail before launching if the optional resource-monitor dependency is absent.
    import psutil  # noqa: F401
    started = time.monotonic()
    deadline = started + args.max_seconds
    output = Path(args.output).absolute()
    require(not output.exists(), 'Refusing an existing output directory')
    output.mkdir(parents=True, exist_ok=False)
    try:
        manifest, profile, plan = check_local_inputs(args)
        write_new(output / 'frozen-inputs.json', manifest)
        write_new(output / 'profile.json', profile)
        arguments = {key: str(Path(getattr(args, key)).resolve())
                     for key in ('manifest', 'profile', 'gguf', 'worker', 'prism_source')}
        arguments.update({key: getattr(args, key) for key in
                          ('expected_manifest_sha256', 'expected_profile_sha256',
                           'expected_gguf_sha256', 'expected_worker_sha256')})
        job = {'schema_version': 1, 'output': str(output), 'arguments': arguments,
               'started_monotonic': started, 'deadline_monotonic': deadline,
               'maximum_whole_process_seconds': args.max_seconds, 'supervisor_pid': os.getpid(),
               'runner_sha256': digest(__file__),
               'code_sha256': {name: digest(ROOT / name) for name in SOURCE_FILES},
               'manifest_sha256': manifest['frozen_sha256'], 'runtime_sha': RUNTIME_SHA,
               'workload': plan['workload'], 'context_length': plan['context_length'],
               'threads': plan['threads'], 'gpu_layers': 0,
               'admission': {'factor': FACTOR, 'reserve_seconds': RESERVE},
               'legacy_fp32_cross_entropy_compatible': False}
        write_new(output / 'job.json', job)
        command = [sys.executable, str(Path(__file__).resolve()), '--child-job', str(output / 'job.json')]
        threads = str(plan['threads'])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1',
                   TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                   VECLIB_MAXIMUM_THREADS=threads, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        result = supervise_command(command, output, deadline, env=env)
        result.update(whole_process_elapsed_seconds=time.monotonic() - started,
                      maximum_whole_process_seconds=args.max_seconds,
                      manifest_sha256=manifest['frozen_sha256'], legacy_fp32_cross_entropy_compatible=False)
        if result['status'] == 'complete':
            report = json.loads((output / 'child-status.json').read_bytes())
            require(report['status'] == 'complete' and (output / 'evaluation.json').is_file(),
                    'Final evaluation or child completion receipt absent')
        write_new(output / 'result.json', result)
        print(json.dumps({'status': result['status'], 'output': str(output), 'reason': result['reason']}))
        return 0 if result['status'] == 'complete' else 1
    except BaseException as error:
        write_new(output / 'failure.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}',
            'traceback': traceback.format_exc(), 'whole_process_elapsed_seconds': time.monotonic() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Explicitly authorize the CPU model run')
    parser.add_argument('--child-job', type=Path, help=argparse.SUPPRESS)
    for name in ('manifest', 'profile', 'gguf', 'worker', 'prism-source', 'output'):
        parser.add_argument('--' + name, type=Path)
    for name in ('manifest', 'profile', 'gguf', 'worker'):
        parser.add_argument('--expected-' + name + '-sha256')
    parser.add_argument('--max-seconds', type=float, help='Whole-process cap, at most 10800 seconds')
    args = parser.parse_args()
    if args.child_job:
        return child_main(args.child_job)
    required = ('manifest', 'profile', 'gguf', 'worker', 'prism_source', 'output', 'max_seconds',
                'expected_manifest_sha256', 'expected_profile_sha256',
                'expected_gguf_sha256', 'expected_worker_sha256')
    if not args.run or any(getattr(args, name) is None for name in required):
        parser.error('--run and all explicit local paths, expected SHA256s and --max-seconds are required')
    return supervisor_main(args)


if __name__ == '__main__':
    raise SystemExit(main())
