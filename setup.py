from setuptools import setup, find_packages

print('Found packages:', find_packages())
setup(
    description='Multi-HMR as a package',
    name='multihmr',
    version='0.0',    
    packages=find_packages(),
    install_requires=[
        'torch==2.0.1',
        'trimesh==3.22.3',
        'pyrender==0.1.45',
        'einops==0.6.1',
        'roma',
        'pillow==10.0.1',
        'smplx',
        'pyvista==0.42.3',
        'numpy==1.22.4',
        'pyglet==1.5.24',
        'tqdm==4.65.0',
        'xformers==0.0.20',

        # for huggingface
        'gradio==4.44.1',
        'spaces==0.19.4',


        # for ehf
        'plyfile==1.0.3',

        # for smpl
        'chumpy==0.70',
    ],
    extras_require={
        'all': [
            
        ],
    },
)
