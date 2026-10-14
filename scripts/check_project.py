#!/usr/bin/env python3
"""Check first-party Python/shell/YAML syntax and maintained documentation links."""
import ast
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = subprocess.run(
        ['git', '-c', f'safe.directory={ROOT}', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
        cwd=ROOT, capture_output=True, check=True,
    )
    names = sorted(set(result.stdout.decode().split('\0')) - {''})
    errors = []
    counts = {'python': 0, 'shell': 0, 'yaml': 0, 'links': 0}
    for name in names:
        path = ROOT / name
        if not path.is_file():
            continue  # Includes removed files still present in the Git index.
        try:
            if path.suffix == '.py':
                ast.parse(path.read_text(), filename=name)
                counts['python'] += 1
            elif path.suffix == '.sh':
                result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
                if result.returncode:
                    errors.append(result.stderr.strip())
                counts['shell'] += 1
            elif path.suffix in {'.yaml', '.yml'}:
                yaml.safe_load(path.read_text())
                counts['yaml'] += 1
        except (SyntaxError, UnicodeError, yaml.YAMLError) as exc:
            errors.append(f'{name}: {exc}')
    submodules = subprocess.run(
        ['git', '-c', f'safe.directory={ROOT}', 'ls-files', '--stage'],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    unavailable = [ROOT / line.split('\t', 1)[1]
                   for line in submodules.stdout.splitlines()
                   if line.startswith('160000 ') and not (ROOT / line.split('\t', 1)[1] / '.git').exists()]
    docs = [ROOT / 'README.md', *sorted((ROOT / 'docs').glob('*.md')),
            ROOT / 'ros2_ws/src/future_risk_vqa/README.md',
            *sorted((ROOT / 'ros2_ws/src/future_risk_vqa/docs').glob('*.md')),
            ROOT / 'robopoint_runtime/README.md']
    for path in docs:
        for target in re.findall(r'\[[^\]]*\]\(([^\s)]+)\)', path.read_text()):
            if re.match(r'[a-zA-Z][a-zA-Z0-9+.-]*:', target) or target.startswith('#'):
                continue
            target = unquote(target.split('#', 1)[0])
            counts['links'] += 1
            if not (path.parent / target).exists():
                resolved = (path.parent / target).resolve()
                if any(resolved.is_relative_to(module.resolve()) for module in unavailable):
                    print(f'Link requires uninitialized submodule: {target}')
                else:
                    errors.append(f'{path.relative_to(ROOT)}: missing link target {target}')
    print('Static checks: ' + ', '.join(f'{key}={value}' for key, value in counts.items()))
    for error in errors:
        print(error)
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
