"""Own one evaluation child and its descendants under an external deadline.

POSIX uses a new process group. Windows uses a non-inheritable kill-on-close
Job Object, assigning a suspended process before resuming its primary thread.
There is no weaker Windows fallback. CPython 3.12+ and the evaluation extra
(psutil) are required. Native Windows harmless-process tests remain necessary;
mock WinAPI tests on another OS are not Windows runtime validation.

This is operational containment for our cooperating evaluator, not a sandbox:
POSIX children must not escape their group. Host failure/SIGKILL cannot produce
a final receipt; on Windows abrupt death in the create-before-assignment gap
can leave a suspended child, but no evaluation code has run in it. Assigned
Windows descendants are killed when the supervisor's sole job handle closes.
"""
from __future__ import annotations

from contextlib import ExitStack
import ctypes
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

_SAMPLE_SECONDS = .25
_LOG_LIMIT = 4 << 20


def _write_json(path, value, *, replace=False):
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    temporary = Path(str(path) + '.tmp') if replace else Path(path)
    with temporary.open('xb') as stream:
        stream.write(encoded)
    if replace:
        os.replace(temporary, path)


def _posix_members(group):
    import psutil
    members = []
    for process in psutil.process_iter(['pid', 'status']):
        try:
            if os.getpgid(process.pid) == group and process.info['status'] != psutil.STATUS_ZOMBIE:
                members.append(process.pid)
        except (ProcessLookupError, PermissionError, psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return members


class _PosixOwner:
    mechanism = 'posix-process-group'

    def __init__(self, command, stdout, stderr, env, deadline):
        if time.monotonic() >= deadline:
            raise TimeoutError('Deadline elapsed before process creation')
        self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                        stdout=stdout, stderr=stderr, env=env,
                                        start_new_session=True)
        self.pid = self.process.pid

    def poll(self):
        return self.process.poll()

    def members(self):
        return _posix_members(self.pid)

    def cleanup(self, deadline):
        expired = time.monotonic() >= deadline
        # Kill at the time boundary before potentially slow process enumeration.
        if expired:
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        before = self.members()
        if before and not expired:
            try:
                os.killpg(self.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            until = min(deadline, time.monotonic() + 1.)
            while time.monotonic() < until and self.members():
                time.sleep(min(.025, max(0., until - time.monotonic())))
            if self.members():
                try:
                    os.killpg(self.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        # This bounded wait observes termination; it grants no extra model time.
        until = time.monotonic() + 2.
        while self.members() and time.monotonic() < until:
            time.sleep(.025)
        try:
            self.process.wait(timeout=max(.01, until - time.monotonic()))
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=1.)
        after = self.members()
        return {'mechanism': self.mechanism, 'pids_before_cleanup': before,
                'live_pids_after_cleanup': after, 'no_live_owned_processes': not after,
                'immediate_deadline_kill': expired, 'ownership_handle_closed': True}


# Explicit Windows widths also make layout tests meaningful on non-Windows hosts.
class _BasicLimits(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64),
                ('PerJobUserTimeLimit', ctypes.c_int64), ('LimitFlags', ctypes.c_uint32),
                ('MinimumWorkingSetSize', ctypes.c_size_t), ('MaximumWorkingSetSize', ctypes.c_size_t),
                ('ActiveProcessLimit', ctypes.c_uint32), ('Affinity', ctypes.c_size_t),
                ('PriorityClass', ctypes.c_uint32), ('SchedulingClass', ctypes.c_uint32)]


class _IOCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        'ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
        'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [('BasicLimitInformation', _BasicLimits), ('IoInfo', _IOCounters),
                ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]


class _WindowsAPI:
    """Narrow Win32 calls; no imports/loading of this API on POSIX.

    Contracts: Microsoft Job Objects / SetInformationJobObject /
    QueryInformationJobObject / ResumeThread; CPython 3.12 _winapi.CreateProcess
    returns both process and thread handles (unlike subprocess.Popen).
    """
    def __init__(self):
        if os.name != 'nt':
            raise RuntimeError('Native Windows APIs are available only on Windows')
        import _winapi
        self.winapi = _winapi
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        declarations = {
            'CreateJobObjectW': ([ctypes.c_void_p, ctypes.c_wchar_p], ctypes.c_void_p),
            'SetInformationJobObject': ([ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32], ctypes.c_int),
            'AssignProcessToJobObject': ([ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int),
            'ResumeThread': ([ctypes.c_void_p], ctypes.c_uint32),
            'TerminateJobObject': ([ctypes.c_void_p, ctypes.c_uint32], ctypes.c_int),
            'QueryInformationJobObject': ([ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                           ctypes.c_uint32, ctypes.c_void_p], ctypes.c_int),
        }
        for name, (arguments, result) in declarations.items():
            function = getattr(self.kernel, name)
            function.argtypes, function.restype = arguments, result

    def _check(self, result):
        if not result:
            raise ctypes.WinError(ctypes.get_last_error())
        return result

    def create_job(self):
        # Null security attributes make the unnamed job handle non-inheritable.
        job = self._check(self.kernel.CreateJobObjectW(None, None))
        try:
            limits = _ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
            self._check(self.kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
        except BaseException:
            self.close(job)
            raise
        return job

    def create_suspended(self, command, stdout, stderr, env):
        import msvcrt
        with open(os.devnull, 'rb') as stdin:
            handles = [msvcrt.get_osfhandle(stream.fileno()) for stream in (stdin, stdout, stderr)]
            startup = subprocess.STARTUPINFO()
            startup.dwFlags = subprocess.STARTF_USESTDHANDLES
            startup.hStdInput, startup.hStdOutput, startup.hStdError = handles
            startup.lpAttributeList = {'handle_list': handles}
            changed = []
            created = None
            try:
                try:
                    for handle in handles:
                        os.set_handle_inheritable(handle, True)
                        changed.append(handle)
                    # CPython supplies EXTENDED_STARTUPINFO_PRESENT and Unicode env.
                    flags = 0x00000004 | 0x08000000  # CREATE_SUSPENDED | CREATE_NO_WINDOW
                    created = self.winapi.CreateProcess(
                        command[0], subprocess.list2cmdline(command), None, None,
                        True, flags, env, None, startup)
                finally:
                    for handle in changed:
                        os.set_handle_inheritable(handle, False)
            except BaseException:
                # A post-creation handle-reset failure must not lose the suspended
                # process handles before the owner can assign it to the job.
                if created is not None:
                    process, thread, _, _ = created
                    try:
                        self.terminate_process(process)
                        self.wait(process, 2000)
                    finally:
                        self.close(thread)
                        self.close(process)
                raise
            process, thread, pid, _ = created
            return process, thread, pid

    def assign(self, job, process):
        self._check(self.kernel.AssignProcessToJobObject(job, process))

    def resume(self, thread):
        previous = self.kernel.ResumeThread(thread)
        if previous != 1:
            if previous == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
            raise RuntimeError(f'Unexpected primary-thread suspend count: {previous}')

    def members(self, job):
        capacity = 16
        while capacity <= 131072:
            class ProcessIDs(ctypes.Structure):
                _fields_ = [('assigned', ctypes.c_uint32), ('listed', ctypes.c_uint32),
                            ('ids', ctypes.c_size_t * capacity)]
            data = ProcessIDs()
            okay = self.kernel.QueryInformationJobObject(job, 3, ctypes.byref(data), ctypes.sizeof(data), None)
            if okay:
                if data.listed > capacity:
                    raise RuntimeError('Invalid Windows job process count')
                return [int(value) for value in data.ids[:data.listed]]
            if ctypes.get_last_error() != 234:  # ERROR_MORE_DATA
                raise ctypes.WinError(ctypes.get_last_error())
            capacity = max(capacity * 2, data.assigned)
        raise RuntimeError('Windows job process inventory exceeded bound')

    def poll(self, process):
        if self.winapi.WaitForSingleObject(process, 0) == 258:  # WAIT_TIMEOUT
            return None
        return self.winapi.GetExitCodeProcess(process)

    def terminate_job(self, job):
        self._check(self.kernel.TerminateJobObject(job, 1))

    def terminate_process(self, process):
        self.winapi.TerminateProcess(process, 1)

    def wait(self, process, milliseconds):
        return self.winapi.WaitForSingleObject(process, milliseconds)

    def close(self, handle):
        self.winapi.CloseHandle(handle)


class _WindowsOwner:
    mechanism = 'windows-kill-on-close-job'

    def __init__(self, command, stdout, stderr, env, deadline, *, api=None):
        self.api = _WindowsAPI() if api is None else api
        self.job = self.process = self.thread = None
        self.pid = None
        self.returncode = None
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError('Deadline elapsed before Windows process creation')
            self.job = self.api.create_job()
            self.process, self.thread, self.pid = self.api.create_suspended(command, stdout, stderr, env)
            self.api.assign(self.job, self.process)
            if time.monotonic() >= deadline:
                raise TimeoutError('Deadline elapsed before resuming Windows child')
            self.api.resume(self.thread)
            self.api.close(self.thread)
            self.thread = None
        except BaseException:
            # Assignment can fail under a host's incompatible parent job policy.
            # Kill the still-suspended process explicitly; never run uncontained.
            try:
                if self.process is not None:
                    self.api.terminate_process(self.process)
                    self.api.wait(self.process, 2000)
            finally:
                for attribute in ('thread', 'process', 'job'):
                    handle = getattr(self, attribute)
                    if handle is not None:
                        self.api.close(handle)
                        setattr(self, attribute, None)
            raise

    def poll(self):
        if self.process is not None:
            self.returncode = self.api.poll(self.process)
        return self.returncode

    def members(self):
        return self.api.members(self.job)

    def cleanup(self, deadline):
        # Terminate the owned job before querying at any exit, including expiry.
        # No child receives extra execution time for sampling or graceful close.
        before = []
        after = None
        closed = False
        expired = time.monotonic() >= deadline
        try:
            if not expired:
                before = self.members()
            self.api.terminate_job(self.job)
            until = time.monotonic() + 2.
            after = self.members()
            while after and time.monotonic() < until:
                time.sleep(.025)
                after = self.members()
            self.api.wait(self.process, max(0, int((until - time.monotonic()) * 1000)))
            self.poll()
        finally:
            # Non-inheritable and sole job handle: this is also the crash backstop.
            if self.job is not None:
                self.api.close(self.job)
                self.job = None
                closed = True
            if self.process is not None:
                self.api.close(self.process)
                self.process = None
        return {'mechanism': self.mechanism, 'pids_before_cleanup': before,
                'live_pids_after_cleanup': after, 'no_live_owned_processes': after == [],
                'immediate_deadline_kill': expired,
                'ownership_handle_closed': closed}


def supervise(command, output, deadline, *, env=None):
    """Run a cooperating evaluator; return and save its bounded lifecycle receipt.

    ``output`` is an already-created, caller-owned new directory. The helper
    exclusively creates runner.stdout, runner.stderr, rss-samples.jsonl and
    supervisor-process.json, and never overwrites existing reserved paths.
    The absolute monotonic deadline includes any caller work already elapsed.
    Cleanup observation may finish after it; model execution receives no grace.
    """
    if not isinstance(command, list) or not command or any(not isinstance(item, str) or not item or '\0' in item for item in command):
        raise ValueError('command must be a nonempty list of nonempty strings')
    if not Path(command[0]).is_absolute():
        raise ValueError('Use an absolute executable path')
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError('deadline must be a finite monotonic timestamp')
    if threading.current_thread() is not threading.main_thread():
        raise ValueError('Supervisor must own signal handlers on the main thread')
    output = Path(output)
    if output.is_symlink() or not output.is_dir():
        raise ValueError('output must be a real, already-created owned directory')
    if os.name not in ('posix', 'nt'):
        raise RuntimeError('Only POSIX and Windows process ownership are supported')
    import psutil
    reserved = ('runner.stdout', 'runner.stderr', 'rss-samples.jsonl',
                'supervisor-process.json', 'supervisor-process.json.tmp')
    if any((output / name).exists() or (output / name).is_symlink() for name in reserved):
        raise FileExistsError('Refusing existing supervisor outputs')
    started = time.monotonic()
    initial = {'status': 'starting', 'command': command, 'deadline_monotonic': deadline,
               'supervisor_pid': os.getpid(), 'platform': os.name,
               'ownership': _WindowsOwner.mechanism if os.name == 'nt' else _PosixOwner.mechanism,
               'sample_interval_seconds': _SAMPLE_SECONDS, 'psutil_version': psutil.__version__,
               'memory_method': 'sampled sum of owned child RSS; shared pages may be counted multiple times; not exact physical-memory peak',
               'native_windows_validation': 'requires harmless-process tests on the target Windows host'}
    report_path = output / 'supervisor-process.json'
    _write_json(report_path, initial)
    owner = None
    reason = None
    exit_code = None
    observed = {}
    peak = samples = supervisor_peak = 0
    handlers = {}

    def interrupted(signum, _frame):
        raise InterruptedError(f'Supervisor received signal {signum}')

    try:
        for name in ('SIGTERM', 'SIGINT', 'SIGHUP', 'SIGBREAK'):
            signum = getattr(signal, name, None)
            if signum is not None:
                handlers[signum] = signal.signal(signum, interrupted)
        with ExitStack() as stack:
            stdout = stack.enter_context((output / 'runner.stdout').open('xb'))
            stderr = stack.enter_context((output / 'runner.stderr').open('xb'))
            memory = stack.enter_context((output / 'rss-samples.jsonl').open('x'))
            factory = _WindowsOwner if os.name == 'nt' else _PosixOwner
            owner = factory(command, stdout, stderr, env, deadline)
            initial.update(status='running', pid=owner.pid)
            _write_json(report_path, initial, replace=True)
            while True:
                now = time.monotonic()
                if now >= deadline:
                    reason = 'whole-process-deadline'
                    break
                rss, rows = 0, []
                for pid in owner.members():
                    try:
                        process = psutil.Process(pid)
                        value, birth = process.memory_info().rss, process.create_time()
                        rss += value
                        key = f'{pid}:{birth}'
                        observed[key] = {'pid': pid, 'create_time': birth,
                                         'max_sampled_rss_bytes': max(value, observed.get(key, {}).get('max_sampled_rss_bytes', 0))}
                        rows.append({'pid': pid, 'rss_bytes': value})
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                peak = max(peak, rss)
                supervisor_peak = max(supervisor_peak, psutil.Process().memory_info().rss)
                samples += 1
                memory.write(json.dumps({'elapsed_seconds': now - started, 'summed_rss_bytes': rss,
                                         'processes': rows}) + '\n')
                memory.flush()
                if stdout.tell() > _LOG_LIMIT or stderr.tell() > _LOG_LIMIT:
                    reason = 'runner-log-limit'
                    break
                exit_code = owner.poll()
                if exit_code is not None:
                    reason = 'exited'
                    break
                time.sleep(min(_SAMPLE_SECONDS, max(0., deadline - time.monotonic())))
    except BaseException as error:
        reason = f'supervisor-error:{type(error).__name__}:{error}'
    finally:
        # A second interrupt must not abandon cleanup of the already-owned tree.
        for signum in handlers:
            signal.signal(signum, signal.SIG_IGN)
        cleanup = {'no_live_owned_processes': owner is None,
                   'live_pids_after_cleanup': [], 'ownership_handle_closed': owner is None}
        try:
            if owner is not None:
                cleanup = owner.cleanup(deadline)
                if exit_code is None:
                    exit_code = owner.poll()
        except BaseException as error:
            cleanup = {'no_live_owned_processes': False, 'ownership_handle_closed': False,
                       'error': f'{type(error).__name__}: {error}'}
        result = {**initial, 'status': 'complete' if reason == 'exited' and exit_code == 0
                  and cleanup['no_live_owned_processes'] else 'failed',
                  'reason': reason, 'exit_code': exit_code, 'cleanup': cleanup,
                  'elapsed_seconds': time.monotonic() - started,
                  'sampled_peak_summed_rss_bytes': peak,
                  'supervisor_sampled_peak_rss_bytes': supervisor_peak,
                  'samples': samples, 'observed_processes': list(observed.values())}
        try:
            _write_json(report_path, result, replace=True)
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
    return result
