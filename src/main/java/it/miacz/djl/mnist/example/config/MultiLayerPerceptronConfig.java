package it.miacz.djl.mnist.example.config;

public record MultiLayerPerceptronConfig(int inputSize,
                                         int outputSize,
                                         int... hiddenLayersSize) {
}
