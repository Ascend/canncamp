
INCLUDE=-I../include \
	-I/usr/local/include/opencv4 \
	-I/home/HwHiAiUser/ascend_ddk/arm/include/opencv4 \
	-I/home/HwHiAiUser/Ascend/ascend-toolkit/3.3.0.alpha001/arm64-linux/acllib/include \
	-I/home/HwHiAiUser/ascend_ddk/arm/include


LIBRARY=$(LD_LIBRARY_PATH) \
	/home/HwHiAiUser/ascend_ddk/arm/lib \
	/usr/local/lib 


LIBS=-L/home/HwHiAiUser/Ascend/ascend-toolkit/latest/arm64-linux/acllib/lib64/stub -lascendcl -lacl_dvpp -lacl_cblas -lacl_retr \
	-L/home/HwHiAiUser/ascend_ddk/arm/lib -latlasutil \
	-L/home/HwHiAiUser/ascend_ddk/arm/lib -lopencv_core -lopencv_flann -lopencv_imgproc -lopencv_ml -lopencv_photo -lopencv_dnn -lopencv_features2d -lopencv_gapi -lopencv_imgcodecs -lopencv_videoio -lopencv_calib3d -lopencv_highgui -lopencv_objdetect -lopencv_stitching -lopencv_video


#编译器
#CC=g++
CC=/usr/bin/aarch64-linux-gnu-g++
CFLAGS=-g -Wall -DENABLE_DVPP_INTERFACE -fPIC 
TARGET=main

#源代码
SRCPPS=main.cpp 

#目标文件
#OBJS = main.o colorization.o
OBJS = main.o 

all: main

clean:
	rm -f *.o

.cpp.o:
	$(CC) $(INCLUDE) $(CFLAGS) -c $<


main: $(OBJS)
	$(CC) -o ../bin/main $(OBJS) $(LIBS)



