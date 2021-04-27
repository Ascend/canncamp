#pragma once

#include <vector>
#include <numeric>
#include <stdlib.h>

using std::vector;

class CProcessOutputTensor
{
public:
    CProcessOutputTensor( const vector<int>& shape, const vector<float>& data )
        : m_shape( shape ), m_data( data ), m_page_size( shape.at(2) * shape.at(3) )
    {
      int size = 1;
      for( auto num : m_shape )
      {
        size *= num;
      }
      if ( size != (int)data.size() )
      {
        LOG_INFO( "error!!: size != data.size()! size:%d, data.size():%zu\n", size, data.size() );
        exit(1);
      }
    };
    ~CProcessOutputTensor() {};

    const std::vector<int>& shape( void ) const { return m_shape; };
    const float* data( void ) const { return m_data.data(); };

    float value_at( int batch, int channel, int height, int width ) const
    {
        int index = channel * m_page_size + m_shape.back() * height + width;
        return data()[index];
    }
private:
    vector<int> m_shape;
    vector<float> m_data;
    const int m_page_size{0};
};

