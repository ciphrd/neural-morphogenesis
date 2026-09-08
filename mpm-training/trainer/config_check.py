"""Verify grouped defaults resolve identically in Python and the browser."""
import json
from pathlib import Path
import subprocess

from config import CONFIG, DEFAULT_RUN_SETTINGS
from chemical_channels import default_channel_profiles, profiles_to_wire


def main():
    root = Path(__file__).resolve().parents[1]
    script = r'''
const fs = require('fs');
const ts = require('./viewer/node_modules/typescript');
require.extensions['.ts'] = (module, filename) => {
  const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {compilerOptions: {
    module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020,
    esModuleInterop: true, resolveJsonModule: true,
  }}).outputText;
  module._compile(compiled, filename);
};
process.stdout.write(JSON.stringify(require('./viewer/src/net/settingsStorage.ts').DEFAULT_RUN_SETTINGS));
'''
    result = subprocess.run(['node', '-e', script], cwd=root, check=True,
                            capture_output=True, text=True)
    browser = json.loads(result.stdout)
    expected = {**DEFAULT_RUN_SETTINGS, 'chemicalChannelProfiles': profiles_to_wire(
        default_channel_profiles(DEFAULT_RUN_SETTINGS['channels']))}
    assert browser == expected, 'Python/browser runtime defaults differ'
    assert DEFAULT_RUN_SETTINGS['channels'] == len(CONFIG['chemistry']['channels'])
    assert not CONFIG['run'].keys() & CONFIG['chemistry'].keys(), 'Duplicate default settings'
    grid = CONFIG['simulation']
    assert grid['DX'] == 1 / grid['GRID_N']
    assert grid['INV_DX'] == grid['GRID_N']
    print('[PASS] grouped chemistry defaults and derived channel count match Python/browser runtime settings')


if __name__ == '__main__':
    main()
