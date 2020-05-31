all:
	python setup.py build_ext --inplace
	rm -rf build

clean:
	rm -rf *.so
	rm -rf *.pyc
	rm rotate_polygon_nms.cpp
