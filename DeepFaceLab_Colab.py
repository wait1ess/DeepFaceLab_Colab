'''
1.准备好workplace：
data_dst:       这个目录是用来存放按帧分解的目标视频的图片；
data_src:       这个目录是用来存放你收集的源脸的图片；
model:          人工智能训练的模型,存放的目录, 每次训练他都会重新去读取你上次训练的结果,所以这个一定要保存好,训练较长时间的模型,
                可以很快速的完成对脸部数据的学习.可以合成出相似度极高的视频
data_dst.mp4:   要替换脸部的目标视频
data_src.mp4:   替换素材图片的视频（按帧分解得到）
该部分自己准备好三个文件夹两个视频即可
'''
#挂载谷歌云盘
#点击链接授权，复制授权码，填入框框，然后回车。
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

%cd /content/drive/My Drive/DeepFaceLab/workspace/
%ls

'''
2.安装DeepFaceLab

'''
# 获取DFL源代码v1.1稳定版，置于文件夹DeepFaceLab_Colab中
!git clone -b v1.1 https://github.com/wait1ess/DeepFaceLab_Colab.git

# 进入DeepFaceLab_Colab目录
%cd /content/drive/My Drive/DeepFaceLab/DeepFaceLab_Colab


# 安装Python依赖
!pip install -r requirements_colab.txt
!pip install --upgrade scikit-image

'''
3.根据两个视频按帧分解得到素材图片
'''
# 对源图片
#Src 视频转图片
!python main.py videoed extract-video --input-file ../workspace/data_src.mp4 --output-dir ../workspace/data_src/
#Src 提取脸部图片
!python main.py extract --input-dir ../workspace/data_src --output-dir ../workspace/data_src/aligned --detector s3fd --debug-dir ../workspace/data_src/aligned_debug
#Src排序，可以通过谷歌云盘查看结果，删除不良图片
!python main.py sort --input-dir ../workspace/data_src/aligned --by hist

# 对目标图片
#Dst视频转图片
!python main.py videoed extract-video --input-file ../workspace/data_dst.mp4 --output-dir ../workspace/data_dst/
#Dst提取脸部图片
!python main.py extract --input-dir ../workspace/data_dst --output-dir ../workspace/data_dst/aligned --detector s3fd --debug-dir ../workspace/data_dst/aligned_debug
#Dst排序，可以通过谷歌云盘查看结果，删除不良图片
!python main.py sort --input-dir ../workspace/data_dst/aligned --by hist

'''
4.训练模型（多选一）
- 支持H128,SAE,DF, LIAEF128等模型，根据自己的情况选择模型。
- 训练开始是需要配置参数，记得开启预览，其他参数根据自己情况选择，使用默认参数的话直接回车即可。
- 不想训练了可以点击停止，停止时会抛出异常，但是没什么关系。下次可以继续训练
- 如果想要查看history ，先停止训练，然后点击下面第三段代码
'''
# Running trainer. SAE 
!python main.py train --training-data-src-dir ../workspace/data_src/aligned --training-data-dst-dir ../workspace/data_dst/aligned --model-dir ../workspace/model --model SAE --no-preview

# Running trainer  H128
!python main.py train --training-data-src-dir ../workspace/data_src/aligned --training-data-dst-dir ../workspace/data_dst/aligned --model-dir ../workspace/model --model H128 --no-preview

# Running trainer. DF 
!python main.py train --training-data-src-dir ../workspace/data_src/aligned --training-data-dst-dir ../workspace/data_dst/aligned --model-dir ../workspace/model --model DF --no-preview

# Running trainer. LIAEF128 
!python main.py train --training-data-src-dir ../workspace/data_src/aligned --training-data-dst-dir ../workspace/data_dst/aligned --model-dir ../workspace/model --model LIAEF128 --no-preview

# 模型预览请使用独立的colab脚本：/DeepFaceLab_Colab/blob/master/ViewLastHistory.ipynb

'''
5.继续训练
# 当你第二次开始训练，或者掉线之后继续训练时并不需要执行上面所有的步骤。只需要下面简单的几个步骤：
5.1 挂载云盘
5.2 安装依赖
5.3 开始训练
'''
#挂载谷歌云盘
#点击链接授权，复制授权码，填入框框，然后回车。

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# 进入DeepFaceLab_Colab目录
%cd /content/drive/My Drive/DeepFaceLab/DeepFaceLab_Colab

# 安装Python依赖
!pip install -r requirements_colab.txt
!pip install --upgrade scikit-image

# 开始训练SAE ，如果是其他模型，修改后面的参数即可。
!python main.py train --training-data-src-dir ../workspace/data_src/aligned --training-data-dst-dir ../workspace/data_dst/aligned --model-dir ../workspace/model --model SAE --no-preview
