using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public abstract class PoolingLayer : Layer2D
    {
        protected int _poolRows, _poolCols;

        protected PoolingLayer()
        {
        }

        protected PoolingLayer(int poolWidth, int poolHeight)
        {
            _poolRows = poolHeight;
            _poolCols = poolWidth;
        }
    }
}
