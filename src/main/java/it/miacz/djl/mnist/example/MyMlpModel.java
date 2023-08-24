package it.miacz.djl.mnist.example;

import ai.djl.Model;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import it.miacz.djl.mnist.example.config.MultiLayerPerceptronConfig;
import it.miacz.djl.mnist.example.config.MyMlpTrainingConfig;
import lombok.Getter;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class MyMlpModel {

    private final MultiLayerPerceptronConfig multiLayerPerceptronConfig;
    @Getter
    private final Model model;

    public MyMlpModel(MultiLayerPerceptronConfig multiLayerPerceptronConfig) {
        this.multiLayerPerceptronConfig = multiLayerPerceptronConfig;
        this.model = Model.newInstance("mlp");
        this.model.setBlock(new Mlp(multiLayerPerceptronConfig.inputSize(), multiLayerPerceptronConfig.outputSize(), multiLayerPerceptronConfig.hiddenLayersSize()));
    }

    public void fit(MyMlpTrainingConfig trainingConfig, Dataset trainingDataset, Dataset validatingDataset) throws TranslateException, IOException {
        model.setProperty("Epochs", String.valueOf(trainingConfig.getNumberOfEpochs()));
        Trainer trainer = getModel().newTrainer(trainingConfig);
        trainer.initialize(new Shape(1, multiLayerPerceptronConfig.inputSize()));
        EasyTrain.fit(trainer, trainingConfig.getNumberOfEpochs(), trainingDataset, validatingDataset);
    }

    public void save(String path) throws IOException {
        Path modelDir = Paths.get(path);
        Files.createDirectories(modelDir);
        model.save(modelDir, model.getName());
        System.out.println(model);
    }

    public Classifications predict(Translator<Image, Classifications> translator, Image img) throws TranslateException {
        try(var predictor = getModel().newPredictor(translator)) {
            return predictor.predict(img);
        }
    }
}
