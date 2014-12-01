default: release

.PHONY: default release debug all clean

PROJECT_NAME=example

include make-utils/flags.mk
include make-utils/cpp-utils.mk

CXX_FLAGS += -Idll/include -Idll/nice_svm/include -Idll/etl/include -std=c++1y
LD_FLAGS  += -lsvm -lopencv_core -lopencv_imgproc -lopencv_highgui -ljpeg -lpthread

ifneq (,$(findstring clang,$(CXX)))
	CXX_FLAGS += -stdlib=libc++
endif

$(eval $(call auto_folder_compile,src))
$(eval $(call auto_add_executable,$(PROJECT_NAME)))

release: release/bin/$(PROJECT_NAME)
debug: debug/bin/$(PROJECT_NAME)

all: release debug

clean: base_clean

include make-utils/cpp-utils-finalize.mk
