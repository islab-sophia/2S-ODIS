wget 'https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1' -O 'vqgan.ckpt'
wget 'https://heibox.uni-heidelberg.de/f/6ecf2af6c658432c8298/?dl=1' -O 'vqgan_config.yaml'
python vqgan_recompile.py
cd eval