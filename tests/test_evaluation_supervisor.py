"""Harmless native process tests and platform-independent mocked WinAPI controls.

Required Windows host gate before model work:
    python -m pytest -q tests/test_evaluation_supervisor.py
The native tests execute this host's actual ownership implementation. Passing
only mocked Windows cases on POSIX does not establish Windows Job behavior.
"""
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from openbonsai import evaluation_supervisor as s


def _alive(pid):
    import psutil
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _wait_path(path, timeout=5):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if path.is_file():
            try:
                value = json.loads(path.read_text())
                if value.get('status') == 'running':
                    return value
            except json.JSONDecodeError:
                pass
        time.sleep(.025)
    raise AssertionError('No running supervisor receipt')


def test_native_clean_exit_and_exclusive_receipts(tmp_path):
    pytest.importorskip('psutil')
    result = s.supervise([sys.executable, '-c', 'print("harmless")'], tmp_path, time.monotonic() + 10)
    assert result['status'] == 'complete' and result['exit_code'] == 0
    assert result['cleanup']['no_live_owned_processes']
    assert result['cleanup']['ownership_handle_closed']
    assert (tmp_path / 'runner.stdout').read_text().strip() == 'harmless'
    assert json.loads((tmp_path / 'supervisor-process.json').read_text()) == result
    with pytest.raises(FileExistsError):
        s.supervise([sys.executable, '-c', 'pass'], tmp_path, time.monotonic() + 5)


def test_native_deadline_kills_child_and_descendant(tmp_path):
    pytest.importorskip('psutil')
    pid_file = tmp_path / 'descendant.json'
    code = ('import subprocess,sys,time,json,pathlib;'
            'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]);'
            f'pathlib.Path({str(pid_file)!r}).write_text(json.dumps({{"pid":p.pid}}));'
            'time.sleep(30)')
    result = s.supervise([sys.executable, '-c', code], tmp_path, time.monotonic() + 1.5)
    assert result['status'] == 'failed' and result['reason'] == 'whole-process-deadline'
    assert result['cleanup']['no_live_owned_processes']
    assert result['elapsed_seconds'] < 6
    assert pid_file.exists()
    assert not _alive(json.loads(pid_file.read_text())['pid'])
    assert not _alive(result['pid'])
    assert result['sampled_peak_summed_rss_bytes'] > 0


def test_native_parent_exit_cleans_orphan_descendant(tmp_path):
    pytest.importorskip('psutil')
    pid_file = tmp_path / 'descendant.json'
    code = ('import subprocess,sys,time,json,pathlib;'
            'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]);'
            f'pathlib.Path({str(pid_file)!r}).write_text(json.dumps({{"pid":p.pid}}));'
            'time.sleep(.3)')
    result = s.supervise([sys.executable, '-c', code], tmp_path, time.monotonic() + 10)
    assert result['status'] == 'complete' and result['exit_code'] == 0
    assert result['cleanup']['pids_before_cleanup']
    assert result['cleanup']['no_live_owned_processes']
    assert not _alive(json.loads(pid_file.read_text())['pid'])


def test_native_log_limit_fails_without_orphans(tmp_path):
    pytest.importorskip('psutil')
    code = 'import sys,time;sys.stdout.write("x"*(5<<20));sys.stdout.flush();time.sleep(30)'
    result = s.supervise([sys.executable, '-c', code], tmp_path, time.monotonic() + 10)
    assert result['status'] == 'failed' and result['reason'] == 'runner-log-limit'
    assert result['cleanup']['no_live_owned_processes']


def _supervisor_code(output):
    return ('import pathlib,sys,time;from openbonsai.evaluation_supervisor import supervise;'
            f'supervise([sys.executable,"-c","import time;time.sleep(30)"],'
            f'pathlib.Path({str(output)!r}),time.monotonic()+20)')


@pytest.mark.skipif(os.name != 'posix', reason='POSIX signal handler contract')
def test_native_posix_sigterm_cleans_owned_child(tmp_path):
    pytest.importorskip('psutil')
    process = subprocess.Popen([sys.executable, '-c', _supervisor_code(tmp_path)])
    try:
        running = _wait_path(tmp_path / 'supervisor-process.json')
        process.terminate()
        process.wait(timeout=5)
        final = json.loads((tmp_path / 'supervisor-process.json').read_text())
        assert final['status'] == 'failed' and 'InterruptedError' in final['reason']
        assert final['cleanup']['no_live_owned_processes'] and not _alive(running['pid'])
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        path = tmp_path / 'supervisor-process.json'
        if path.exists():
            pid = json.loads(path.read_text()).get('pid')
            if pid and _alive(pid):
                os.killpg(pid, signal.SIGKILL)


@pytest.mark.skipif(os.name != 'nt', reason='Requires actual Windows kill-on-close Job Object')
def test_native_windows_supervisor_death_closes_job(tmp_path):
    pytest.importorskip('psutil')
    process = subprocess.Popen([sys.executable, '-c', _supervisor_code(tmp_path)])
    child = None
    try:
        child = _wait_path(tmp_path / 'supervisor-process.json')['pid']
        process.kill()  # Uncatchable TerminateProcess, not Python's signal cleanup.
        process.wait(timeout=5)
        until = time.monotonic() + 5
        while _alive(child) and time.monotonic() < until:
            time.sleep(.025)
        assert not _alive(child)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        if child and _alive(child):
            import psutil
            psutil.Process(child).kill()


@pytest.mark.parametrize('deadline', [float('inf'), float('nan'), True])
def test_invalid_deadline_never_creates_outputs(tmp_path, deadline):
    with pytest.raises(ValueError, match='finite'):
        s.supervise([sys.executable, '-c', 'pass'], tmp_path, deadline)
    assert not list(tmp_path.iterdir())


def test_elapsed_deadline_never_launches(tmp_path):
    pytest.importorskip('psutil')
    result = s.supervise([sys.executable, '-c', 'raise RuntimeError("must not launch")'],
                         tmp_path, time.monotonic() - 1)
    assert result['status'] == 'failed' and 'TimeoutError' in result['reason']
    assert 'pid' not in result and result['cleanup']['no_live_owned_processes']


@pytest.mark.parametrize('command', [[], ['relative-python'], [sys.executable, ''], 'shell text'])
def test_invalid_command_rejected(tmp_path, command):
    with pytest.raises(ValueError):
        s.supervise(command, tmp_path, time.monotonic() + 5)
    assert not list(tmp_path.iterdir())


def test_existing_log_is_preserved_before_launch(tmp_path, monkeypatch):
    pytest.importorskip('psutil')
    log = tmp_path / 'runner.stdout'
    log.write_text('preserve')
    monkeypatch.setattr(s, '_PosixOwner', lambda *a: pytest.fail('must not launch'))
    with pytest.raises(FileExistsError):
        s.supervise([sys.executable, '-c', 'pass'], tmp_path, time.monotonic() + 5)
    assert log.read_text() == 'preserve'


class FakeWindowsAPI:
    def __init__(self, failure=None, clock=None):
        self.events = []
        self.failure = failure
        self.clock = clock
        self.active = [123, 456]
        self.exited = None

    def create_job(self):
        self.events.append('create-job')
        if self.failure == 'job':
            raise OSError('job failure')
        return 10

    def create_suspended(self, *args):
        self.events.append('create-suspended')
        if self.failure == 'create':
            raise OSError('create failure')
        return 20, 30, 123

    def assign(self, job, process):
        assert (job, process) == (10, 20)
        self.events.append('assign')
        if self.failure == 'assign':
            raise OSError('assignment failure')
        if self.clock is not None:
            self.clock[0] = 30

    def resume(self, thread):
        assert thread == 30
        self.events.append('resume')
        if self.failure == 'resume':
            raise OSError('resume failure')

    def close(self, handle):
        self.events.append(('close', handle))

    def terminate_process(self, process):
        self.events.append(('terminate-process', process))
        self.exited = 1

    def terminate_job(self, job):
        self.events.append(('terminate-job', job))
        self.active = []
        self.exited = 1

    def members(self, job):
        self.events.append(('members', job))
        return self.active[:]

    def wait(self, process, ms):
        self.events.append(('wait', process))
        return 0

    def poll(self, process):
        return self.exited


def test_windows_owner_assigns_before_resume_and_closes_handles():
    api = FakeWindowsAPI()
    owner = s._WindowsOwner(['example'], None, None, None, time.monotonic() + 10, api=api)
    assert api.events == ['create-job', 'create-suspended', 'assign', 'resume', ('close', 30)]
    cleanup = owner.cleanup(time.monotonic() + 10)
    assert cleanup['no_live_owned_processes'] and cleanup['ownership_handle_closed']
    assert api.events[-2:] == [('close', 10), ('close', 20)]
    assert owner.poll() == 1


@pytest.mark.parametrize('failure', ['create', 'assign', 'resume'])
def test_windows_setup_failure_never_leaves_uncontained_process(failure):
    api = FakeWindowsAPI(failure)
    with pytest.raises(OSError):
        s._WindowsOwner(['example'], None, None, None, time.monotonic() + 10, api=api)
    assert ('close', 10) in api.events
    if failure == 'create':
        assert 'assign' not in api.events and 'resume' not in api.events
    else:
        assert ('terminate-process', 20) in api.events
        assert ('close', 20) in api.events and ('close', 30) in api.events
        assert api.events.index(('terminate-process', 20)) < api.events.index(('close', 20))
    if failure == 'assign':
        assert 'resume' not in api.events


def test_windows_deadline_before_resume_kills_suspended_child(monkeypatch):
    clock = [10]
    monkeypatch.setattr(s.time, 'monotonic', lambda: clock[0])
    api = FakeWindowsAPI(clock=clock)
    with pytest.raises(TimeoutError, match='resuming'):
        s._WindowsOwner(['example'], None, None, None, 20, api=api)
    assert 'resume' not in api.events and ('terminate-process', 20) in api.events


def test_windows_expired_cleanup_terminates_before_enumeration():
    api = FakeWindowsAPI()
    owner = s._WindowsOwner(['example'], None, None, None, time.monotonic() + 10, api=api)
    api.events.clear()
    result = owner.cleanup(time.monotonic() - 1)
    assert api.events[0] == ('terminate-job', 10)
    assert result['immediate_deadline_kill']


def test_windows_extended_limit_layout_and_flags():
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        assert ctypes.sizeof(s._BasicLimits) == 64
        assert ctypes.sizeof(s._ExtendedLimits) == 144
    captured = {}
    def create(security, name):
        assert security is None and name is None
        return 10
    def configure(job, kind, pointer, size):
        assert job == 10 and kind == 9 and size == ctypes.sizeof(s._ExtendedLimits)
        limits = ctypes.cast(pointer, ctypes.POINTER(s._ExtendedLimits)).contents
        captured['flags'] = limits.BasicLimitInformation.LimitFlags
        return 1
    api = s._WindowsAPI.__new__(s._WindowsAPI)
    api.kernel = SimpleNamespace(CreateJobObjectW=create, SetInformationJobObject=configure)
    assert api.create_job() == 10 and captured['flags'] == 0x2000


def test_windows_create_suspended_restricts_handles_and_preserves_argv(monkeypatch, tmp_path):
    events, captured = [], {}
    monkeypatch.setitem(sys.modules, 'msvcrt', SimpleNamespace(get_osfhandle=lambda fd: fd))
    monkeypatch.setattr(subprocess, 'STARTUPINFO', SimpleNamespace, raising=False)
    monkeypatch.setattr(subprocess, 'STARTF_USESTDHANDLES', 0x100, raising=False)
    monkeypatch.setattr(os, 'set_handle_inheritable', lambda h, v: events.append((h, v)), raising=False)
    def create(*args):
        captured['args'] = args
        return 20, 30, 123, 321
    api = s._WindowsAPI.__new__(s._WindowsAPI)
    api.winapi = SimpleNamespace(CreateProcess=create)
    command = ['C:\\Python\\python.exe', '-c', 'print("space argument")']
    with (tmp_path / 'out').open('xb') as stdout, (tmp_path / 'err').open('xb') as stderr:
        assert api.create_suspended(command, stdout, stderr, {'TEST': 'value'}) == (20, 30, 123)
    args = captured['args']
    assert args[0] == command[0] and args[1] == subprocess.list2cmdline(command)
    assert args[4] is True and args[5] & 4 and not args[5] & 0x01000000  # no breakaway
    assert args[6] == {'TEST': 'value'}
    startup = args[-1]
    assert startup.lpAttributeList['handle_list'] == [startup.hStdInput, startup.hStdOutput, startup.hStdError]
    assert len(events) == 6 and all(flag for _, flag in events[:3])
    assert all(not flag for _, flag in events[3:])
