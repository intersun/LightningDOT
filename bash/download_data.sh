cur_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
home_dir="$(dirname $cur_dir)"

cd $home_dir
mkdir data
mkdir data/model
mkdir data/db
mkdir data/img
mkdir data/meta

cd data/model/
# download all pre-trained model checkpoints
wget -nc https://acvrpublicycchen.blob.core.windows.net/lightningdot/LightningDot.pt
wget -nc https://acvrpublicycchen.blob.core.windows.net/lightningdot/bert-base-cased.pt
wget -nc https://acvrpublicycchen.blob.core.windows.net/lightningdot/uniter-base.pt
wget -nc https://acvrpublicycchen.blob.core.windows.net/lightningdot/coco-ft.pt
wget -nc https://acvrpublicycchen.blob.core.windows.net/lightningdot/flickr-ft.pt

cd ../meta/
# download meta files for both coco and flickr
wget -nc https://acvrpublicycchen.blob.core.windows.net/lightningdot/coco_meta.json
wget -nc https://acvrpublicycchen.blob.core.windows.net/lightningdot/flickr_meta.json

