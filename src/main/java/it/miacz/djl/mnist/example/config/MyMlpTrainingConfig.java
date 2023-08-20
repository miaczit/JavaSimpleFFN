package it.miacz.djl.mnist.example.config;

import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import lombok.Getter;

public class MyMlpTrainingConfig extends DefaultTrainingConfig {

    @Getter
    private final int numberOfEpochs;

    public MyMlpTrainingConfig(Loss loss, Evaluator evaluator, TrainingListener[] trainingListeners, int numberOfEpochs) {
        super(loss);
        this.addEvaluator(evaluator);
        this.addTrainingListeners(trainingListeners);
        this.numberOfEpochs = numberOfEpochs;
    }
}
