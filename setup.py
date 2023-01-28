from setuptools import setup, find_packages

setup(
    name='wandb_tools',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
        'wandb'
    ],
    # entry_points={
    #     'console_scripts': [
    #         'yourscript = app.yourscript:cli',
    #     ],
    # },
)