# BGGAN: Bokeh-Glass Generative Adversarial Network for Rendering Realistic Bokeh
## AIM 2020 Challenge on Rendering Realistic Bokeh
## Both track 1 & track 2 rank first
[Competition Link](https://competitions.codalab.org/competitions/24716#learn_the_details)


* Full Results Link：https://pan.baidu.com/s/1U83LuXL6UmwYtGDzYaHwkw   passwd：WJ45
* pretrained model & test image https://drive.google.com/drive/folders/1xnUTUHphKrrDN3MFJaiwDnN7XYUPq1eM?usp=sharing

## Use our code

Before you run this model,you should install the following packges:

python >= 3.6 

tensorflow=2.2.0 

argparse 

then you should run the model like the following command: 
CUDA_VISIBLE_DEVICES='0' python main.py --result_path ./result 

you can change your CUDA devices id and the path to save result images. 

------------------------------------------------------------------------------

Our paper accepted by eccv workshop.  
If you find our paper is useful for you ,
please cite us:  
@inproceedings{qian2020bggan,  
  title={Bggan: Bokeh-glass generative adversarial network for rendering realistic bokeh},  
  author={Qian, Ming and Qiao, Congyu and Lin, Jiamin and Guo, Zhenyu and Li, Chenghua and Leng, Cong and Cheng, Jian},  
  booktitle={European Conference on Computer Vision},   
}  
If you have any issues, please contact mingqian@whu.edu.cn.


------------------------------------------------------------------------------
Our code can traslate to tflite file in a simple way.
If you wanna reproduce our code, you can use the image pair list in "new_list.txt" for training.
