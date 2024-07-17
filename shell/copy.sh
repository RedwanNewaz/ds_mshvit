ROOT_DIR="/home/aredwann/MSHViT"

copyFile()
{
  cp "$ROOT_DIR/$1.py" ../core
}


#copyFile backbones
#copyFile heads
#copyFile blocks
#copyFile dataloader
#copyFile transforms
#copyFile layers
#copyFile tokenizers
#copyFile datamodules
#copyFile ema

#for net in coatnet tresnet botnet
#do
#  copyFile $net
#done

#copyFile metrics

for net in optimizer scheduler class_weight ema
do
  copyFile $net
done