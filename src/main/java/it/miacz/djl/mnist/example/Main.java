package it.miacz.djl.mnist.example;

import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import it.miacz.djl.mnist.example.config.MultiLayerPerceptronConfig;
import it.miacz.djl.mnist.example.config.MyMlpTrainingConfig;

import java.io.IOException;

public class Main {

    public static void main(String[] args) throws IOException, TranslateException {
        var singleImageWidth = 28;
        var singleImageHeight = 28;
        var numberOfClasses = 10;
        int numberOfEpochs = 2;

        MultiLayerPerceptronConfig multiLayerPerceptronConfig = new MultiLayerPerceptronConfig( singleImageWidth*singleImageHeight, numberOfClasses, 128, 64);
        MyMlpModel myMlpModel = new MyMlpModel(multiLayerPerceptronConfig);

        MyMlpTrainingConfig trainingConfig = new MyMlpTrainingConfig(
                Loss.softmaxCrossEntropyLoss(),
                new Accuracy(),
                TrainingListener.Defaults.logging(),
                numberOfEpochs);

        int batchSize = 32;
        Mnist mnist = prepareDataset(batchSize);

        myMlpModel.fit(trainingConfig, mnist, null);

        var img = ImageFactory.getInstance().fromUrl("https://resources.djl.ai/images/0.png");
        img.getWrappedImage();

        Translator<Image, Classifications> translator = new SimpleImageDataTranslator(multiLayerPerceptronConfig);
        System.out.println(myMlpModel.predict(translator, img));
    }

    private static Mnist prepareDataset(int batchSize) throws IOException {
        Mnist mnist = Mnist.builder().setSampling(batchSize, true).build();
        mnist.prepare(new ProgressBar());
        return mnist;
    }
}
