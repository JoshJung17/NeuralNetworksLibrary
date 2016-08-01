using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public class Neuron
    {
        internal float Value;
        internal float Derivative;
        private Neuron[] _inputNeurons;
        private WeightSet _weights;
        private Neuron _imitatedNeuron;
        private Func<float, float> _activation;
        private Func<float, float> _derivative;
        private bool _dead;

        internal Neuron()
        {

        }

        internal Neuron(Neuron[] inputNeurons, WeightSet weights, int activationFunction)
        {
            _inputNeurons = inputNeurons;
            _weights = weights;

            if (activationFunction == NeuralNetwork2D.ACTIVATION_TANH)
            {
                _activation = TanhActivation;
                _derivative = TanhDerivative;
            }
            else if (activationFunction == NeuralNetwork2D.ACTIVATION_RELU)
            {
                _activation = ReLUActivation;
                _derivative = ReLUDerivative;
            }
        }

        internal void Imitate(Neuron prevNeuron)
        {
            _imitatedNeuron = prevNeuron;
            Value = prevNeuron.Value;
        }

        internal void UpdateValue()
        {
            _dead = false;
            float weightedSum = _weights.EvaluateWith(_inputNeurons);
            Value = _activation(weightedSum);
        }

        internal void BackProp()
        {
            if (_dead)
            {
                Derivative = 0;
                return;
            }
            if (Derivative == 0) return;
            if (_imitatedNeuron != null)
            {
                _imitatedNeuron.Derivative = Derivative;
                Derivative = 0;
                return;
            }
            Derivative *= _derivative(Value);
            _weights.BackProp(_inputNeurons, Derivative);
        }

        internal void Die()
        {
            Value = 0;
            _dead = true;
        }

        internal void UpdateWeights(float learningRate)
        {
            if (Derivative == 0) return;
            _weights.UpdateWeights(_inputNeurons, Derivative, learningRate);
            Derivative = 0;
        }

        private float TanhActivation(float weightedSum)
        {
            return (float)(1.71*Math.Tanh(weightedSum));
        }

        private float ReLUActivation(float weightedSum)
        {
            return weightedSum < 0 ? 0.01f * weightedSum : weightedSum;
        }

        private float TanhDerivative(float activationResult)
        {
            return 1.71f - activationResult * activationResult / 1.71f;
        }

        private float ReLUDerivative(float activationResult)
        {
            return activationResult < 0 ? 0.01f : 1;
        }
    }
}
