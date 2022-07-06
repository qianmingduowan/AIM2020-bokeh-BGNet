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

If you wanna reproduce our code, you can use the image pair list in "new_list.txt" for training because we manually clean the train data of EBB! dataset. 

If anyone wants to get my unsorted code in an uncommercial way, you can email me \& I will send it to you. (I didn't have time to sort it, I can send torch or tf2.0 version)


### [License](./LICENSE.md)

All rights reserved.
Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**)

The code is released for academic research use only. For commercial use, please contact [mingqian@whu.edu.cn](mingqian@whu.edu.cn).
