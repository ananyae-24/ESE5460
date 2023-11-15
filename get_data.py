import json
import os
import shutil
import subprocess

if not os.path.isdir("./data"):
    os.makedirs("./data")

if not os.path.isfile('./data/hg38.ml.fa'):
    print('downloading hg38.ml.fa')
    subprocess.call('curl -o ./data/hg38.ml.fa.gz https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz', shell=True)
    subprocess.call('gzip -d ./data/hg38.ml.fa.gz', shell=True)

if not os.path.exists('./data/coolers'):
    os.mkdir('./data/coolers')
if not os.path.isfile('./data/coolers/HFF_hg38_4DNFIP5EUOFX.mapq_30.2048.cool'):
    subprocess.call('curl -o ./data/coolers/H1hESC_hg38_4DNFI1O6IL1Q.mapq_30.2048.cool'+
            ' https://storage.googleapis.com/basenji_hic/tutorials/coolers/H1hESC_hg38_4DNFI1O6IL1Q.mapq_30.2048.cool', shell=True)