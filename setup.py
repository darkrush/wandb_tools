from setuptools import setup, find_packages

setup(
    name='wandb_tools',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
        'wandb'
    ],
    entry_points={
        'console_scripts': [
            'wandb_tools_groups = wandb_tools.app.groups:main',
            'wandb_tools_siblings = wandb_tools.app.siblings:main',
            'wandb_tools_update = wandb_tools.app.update:main',
            'wandb_tools_display_results = wandb_tools.app.display_results:main'
        ]
    }
)