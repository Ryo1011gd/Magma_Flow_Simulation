# ==== Minimal Makefile ====
TARGET := VULCANO

# nvc++(GPU) が既定。CPUでビルドしたい時は: make CPU=1
CPU ?= 0
ifeq ($(CPU),1)
  CXX      := g++
  CXXFLAGS := -std=c++17 -O3 -fopenmp
  LDFLAGS  := -lm -fopenmp
else
  CXX      := nvc++
  GPU_ARCH ?= cc120# 例: cc80 / cc90 / cc120
  CXXFLAGS := -std=c++17 -O3 -Minfo=accel -acc -gpu=$(GPU_ARCH),mem:separate,rdc
  LDFLAGS  := -acc -gpu=$(GPU_ARCH),mem:separate,rdc -lm
endif

SRCS := rigid.cpp errorfunc.cpp log.cpp
OBJS := $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) $(TARGET).exe

# 依存関係（任意だが安全）
main.o:     log.h
errorfunc.o: errorfunc.h
log.o:       errorfunc.h log.h