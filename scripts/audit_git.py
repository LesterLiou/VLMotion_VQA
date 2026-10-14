#!/usr/bin/env python3
"""Read-only audit of this checkout and declared submodules; --remote queries GitHub."""
import argparse
import configparser
from datetime import datetime
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def git(path, *args):
    return subprocess.run(
        ['git', '-c', f'safe.directory={path}', '-C', str(path), *args],
        capture_output=True, text=True, timeout=30
    )


def repositories(path):
    yield path
    modules = path / '.gitmodules'
    if modules.is_file():
        config = configparser.ConfigParser()
        config.read(modules)
        for section in config.sections():
            child = path / config[section]['path']
            yield from repositories(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote', action='store_true', help='Query live remote branch SHAs without fetch/pull')
    args = parser.parse_args()
    print(datetime.now(ZoneInfo('Asia/Taipei')).isoformat(timespec='seconds'))
    failed = False
    for path in repositories(ROOT):
        label = '.' if path == ROOT else str(path.relative_to(ROOT))
        print(f'\n[{label}]', flush=True)
        if not (path / '.git').exists():
            print('UNINITIALIZED: declared submodule; no independent checkout')
            continue
        print(git(path, 'remote', '-v').stdout.rstrip())
        print(git(path, 'branch', '-vv').stdout.rstrip())
        dirty = git(path, 'status', '--short').stdout.rstrip()
        print(dirty or 'Working tree clean')
        if not args.remote:
            print('Remote freshness UNKNOWN (use --remote; local refs may be stale).')
            continue
        head = git(path, 'rev-parse', 'HEAD').stdout.strip()
        for remote in git(path, 'remote').stdout.splitlines():
            try:
                result = git(path, 'ls-remote', '--heads', remote)
            except subprocess.TimeoutExpired:
                print(f'{remote}: TIMEOUT; freshness UNKNOWN')
                failed = True
                continue
            if result.returncode:
                print(f'{remote}: ERROR; freshness UNKNOWN\n{result.stderr.strip()}')
                failed = True
                continue
            heads = {}
            for line in result.stdout.splitlines():
                sha, ref = line.split()
                heads[ref.removeprefix('refs/heads/')] = sha
            for name, sha in sorted(heads.items()):
                marker = ' = current HEAD' if sha == head else ''
                print(f'{remote}/{name}: {sha}{marker}')
            local = git(path, 'for-each-ref', '--format=%(refname:short)|%(objectname)|%(upstream:short)', 'refs/heads').stdout
            for line in local.splitlines():
                branch, sha, upstream = line.split('|')
                if upstream.startswith(remote + '/'):
                    name = upstream[len(remote) + 1:]
                    status = 'REMOTE BRANCH MISSING' if name not in heads else ('MATCH' if sha == heads[name] else 'DIFFERS; fetch and inspect ancestry before updating')
                    print(f'  {branch} -> {upstream}: {status}')
                elif not upstream:
                    print(f'  {branch}: no upstream configured')
    return 2 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
