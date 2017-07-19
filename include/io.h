//
// Created by 芦yafei  on 17/6/23.
//

#ifndef LU_NET_IO_H
#define LU_NET_IO_H

#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include "net.h"
#include "../proto/lu.pb.h"

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace lu_net {
    using google::protobuf::io::FileInputStream;
    using google::protobuf::io::FileOutputStream;
    using google::protobuf::io::ZeroCopyInputStream;
    using google::protobuf::io::CodedInputStream;
    using google::protobuf::io::ZeroCopyOutputStream;
    using google::protobuf::io::CodedOutputStream;
    using google::protobuf::Message;

    /**
    * convert label tag to vector
    **/
    void label2vec(const label_t *t, int num, std::vector<vec_t> &vec);


    void normalize_tensor(const std::vector<tensor_t> &inputs, std::vector<tensor_t> &normalized);


    /**
     * convert vec_t to tensor_t
     **/
    void normalize_tensor(const std::vector<vec_t> &inputs, std::vector<tensor_t> &normalized);


    /**
     * convert label_t to one hot tensor_t
     **/
    void normalize_tensor(const std::vector<label_t> &inputs, std::vector<tensor_t> &normalized);


    bool ReadProtoFromTextFile(const char *filename, Message *proto);


    void WriteProtoToTextFile(const Message &proto, const char *filename);


    bool ReadProtoFromBinaryFile(const char *filename, Message *proto);


    void WriteProtoToBinaryFile(const Message &proto, const char *filename);
}
#endif //LU_NET_IO_H
