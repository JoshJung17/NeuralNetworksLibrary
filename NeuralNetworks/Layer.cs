using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Reflection;

namespace NeuralNetworks
{
    public abstract class Layer
    {
        protected Layer _nextLayer;
        protected Layer _prevLayer;

        protected Layer()
        {
        }

        internal abstract void BindInputFrom(Layer prevLayer, bool randomizeWeights);

        internal abstract float[] Propagate(bool realRun);

        internal abstract float PropagateAndLearn(float[] targetOutput, float learningRate);

        internal abstract void BackPropagate(float learningRate);

        internal abstract Neuron[] GetAllNeurons();

        internal abstract void SaveWeights(StreamWriter writer);

        internal static Layer GetLayerFrom(StreamReader reader)
        {
            Type layerType = Type.GetType(reader.ReadLine());
            ConstructorInfo constructor = layerType.GetConstructor(new Type[1] { typeof(StreamReader) });
            return (Layer)constructor.Invoke(new object[] { reader });
        }
    }
}
