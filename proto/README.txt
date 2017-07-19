Generate lu.pb.h lu.pb.cc for c++ use:
protoc -I=./ --cpp_out=./ ./lu.proto

Generate lu_pb2.py for python use:
protoc -I=./ --python_out=../python/proto/ ./lu.proto