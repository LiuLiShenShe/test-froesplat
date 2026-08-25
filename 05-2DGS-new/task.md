换一个思路，我使用标准版的2DGS：D:\CAAS\2d-gaussian-splatting-main，不使用魔改版。在标准版2DGS目录下创建虚拟环境，安装依赖。查看Readme.md文档了解具体训练和Mesh方法。

使用SAM:D:\CAAS\03-SAM 分割完成的图像进行位姿匹配的操作，应该是标准版的2DGS：D:\CAAS\2d-gaussian-splatting-main 中有脚本。输出在D:\CAAS\04-COLMAP-new

位姿匹配完成后进行训练，加上2DGS特有的几个参数。输出在D:\CAAS\05-2DGS-new。随后再进行黑色废光清洗

训练完成后，进行MESH化，输出在D:\CAAS\06-MESH-new

有问题直接向我提问。