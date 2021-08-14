import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='augmennt',
    version='0.1',
    description='Augmentations for super-resolution, restoration and image-to-image translation CNN models',
    keywords='pytorch image augmentations',
    long_description=long_description, 
    long_description_content_type='text/markdown', 
    url='https://github.com/victorca25/augmennt',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)