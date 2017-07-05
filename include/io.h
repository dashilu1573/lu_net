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
 * label转成向量形式
 **/
    void label2vec(const label_t *t,
                   int num,
                   vector<vec_t> &vec) {
        int outdim = 10;
        vec.reserve(num);
        for (int i = 0; i < num; i++) {
            assert(t[i] < outdim);
            vec.emplace_back(outdim, 0);
            vec.back()[t[i]] = 1;
        }
    }


    void normalize_tensor(const vector<tensor_t> &inputs,
                          vector<tensor_t> &normalized) {
        normalized = inputs;
    }


/**
 * convert vec_t to tensor_t
 **/
    void normalize_tensor(const vector<vec_t> &inputs,
                          vector<tensor_t> &normalized) {
        normalized.reserve(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++)
            //emplace_back和push_back都是向容器内添加数据，emplace_back更高效
            normalized.emplace_back(tensor_t{inputs[i]});
    }


/**
 * convert label_t to one hot tensor_t
 **/
    void normalize_tensor(const vector<label_t> &inputs,
                          vector<tensor_t> &normalized) {
        vector<vec_t> vec;
        normalized.reserve(inputs.size());
        label2vec(&inputs[0], inputs.size(), vec);
        normalize_tensor(vec, normalized);
    }


    bool ReadProtoFromTextFile(const char *filename, Message *proto) {
        int fd = open(filename, O_RDONLY);
        if (-1 != fd) {
            cout << "File not found: " << filename << endl;
        }
        FileInputStream *input = new FileInputStream(fd);
        bool success = google::protobuf::TextFormat::Parse(input, proto);
        delete input;
        close(fd);
        return success;
    }


    void WriteProtoToTextFile(const Message &proto, const char *filename) {
        string str;
        google::protobuf::TextFormat::PrintToString(proto, &str);
        printf("%s", str.c_str());

        int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        FileOutputStream *output = new FileOutputStream(fd);
        google::protobuf::TextFormat::Print(proto, output);
        delete output;
        close(fd);
    }


    bool ReadProtoFromBinaryFile(const char *filename, Message *proto) {
        int fd = open(filename, O_RDONLY);
        if (-1 != fd) {
            cout << "File not found: " << filename << endl;
        }
        ZeroCopyInputStream *raw_input = new FileInputStream(fd);
        CodedInputStream *coded_input = new CodedInputStream(raw_input);
        coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

        bool success = proto->ParseFromCodedStream(coded_input);

        delete coded_input;
        delete raw_input;
        close(fd);
        return success;
    }


    void WriteProtoToBinaryFile(const Message &proto, const char *filename) {
        fstream output(filename, ios::out | ios::trunc | ios::binary);
        proto.SerializeToOstream(&output);
    }
}
#endif //LU_NET_IO_H
