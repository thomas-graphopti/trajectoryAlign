from setuptools import setup, find_packages

setup(
    name='modelAlign',
    version='0.1.0',
    description='A package for aligning models',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/modelAlign',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'iniconfig==2.0.0',
        'numpy==1.26.4',
        'packaging==24.0',
        'pluggy==1.4.0',
        'pymap3d==3.1.0',
        'pytest==8.1.1',
        'scipy==1.12.0'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
