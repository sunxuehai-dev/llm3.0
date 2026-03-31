from dataclasses import fields
from .LlamaConfig import LlamaConfig
from .ConflictCheck import ConfigValidator
import os
import signal
import subprocess
import logging



DEBUG_MODE = True

if DEBUG_MODE:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
else:
    class DummyLogger:
        def info(self, msg): pass
        def error(self, msg): pass
        def debug(self, msg): pass
        def warning(self, msg): pass
    
    logger = DummyLogger()


class FactoryCli:
    def __init__(self, config: LlamaConfig, *, command_type = 'train', command_prefix: str = "") -> None:
        """
        初始化FactoryCli实例，用于构建和执行llamafactory命令
        可以通过 command_prefix来设置llamafactory的路径
        或者多GPU设置
        """
        self._env_var = {}
        self._prefix = command_prefix
        self._config = config
        self._type = command_type
        ConfigValidator(self._config, self._type).validate()

    def _gen_llamafactory_cmd(self) -> None:
        cmd = [f'{self._prefix}/llamafactory-cli', f'{self._type}']
    
        for field in fields(self._config):
            params_name = f'--{field.name}'
            params_value = getattr(self._config, field.name)
            if isinstance(params_value, bool):
                params_value = str(params_value).lower()

            if params_value == '-':
                continue

            if params_value is not None:
                cmd.append(f'{params_name}') 
                cmd.append(f'{str(params_value)}')
        self._cmd = cmd


    def add_env_var(self, key: str, value: str) -> None:
        self._env_var[key] = value

    def server_term(self):
        self.process.terminate()
        self.process.wait()
        self.process.kill()

        process = subprocess.Popen(
            ["killall", "llamafactory-cli"]
        )
        process.wait()


    def server(self):
        self._gen_llamafactory_cmd()
        logger.info(f"Running command: {' '.join(self._cmd)}")
        
        env = os.environ.copy()
        env.update(self._env_var)
        
        self.process = subprocess.Popen(
            self._cmd,
            env=env
        )

    def run(self):
        self._gen_llamafactory_cmd()
        logger.info(f"Running command: {' '.join(self._cmd)}")
        
        env = os.environ.copy()
        env.update(self._env_var)
        
        process = subprocess.Popen(
            self._cmd,
            env=env
        )

        returncode = process.wait()
        return returncode
