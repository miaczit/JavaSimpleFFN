package it.miacz.djl.mnist.example;

import ai.djl.Model;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.nn.Block;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import it.miacz.djl.mnist.example.config.MultiLayerPerceptronConfig;
import it.miacz.djl.mnist.example.config.MyMlpTrainingConfig;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.*;

class MyMlpModelTest {

    private final MultiLayerPerceptronConfig multiLayerPerceptronConfig = new MultiLayerPerceptronConfig(28*28, 10, 128, 64);

    @Test
    void shouldCreateMlpModelBasedOnConfig() {
        MyMlpModel myMlpModel = new MyMlpModel(multiLayerPerceptronConfig);
        Model model = myMlpModel.getModel();
        Block block = model.getBlock();

        assertThat(block).isInstanceOf(Mlp.class);
    }

    @Test
    void shouldExecuteFittingWithGivenConfig() throws TranslateException, IOException {
        //given
        var myMlpModel = new MyMlpModel(multiLayerPerceptronConfig);
        var modelSpy = spy(myMlpModel);
        MyMlpTrainingConfig myMlpTrainingConfig = mock(MyMlpTrainingConfig.class);
        Dataset trainingDataset = mock(Dataset.class);
        var expectedNumberOfEpochs = 2;
        doReturn(expectedNumberOfEpochs).when(myMlpTrainingConfig).getNumberOfEpochs();
        Trainer trainer = mock(Trainer.class);
        Model model = mock(Model.class);
        doReturn(model).when(modelSpy).getModel();
        doReturn(trainer).when(model).newTrainer(any());

        //when
        modelSpy.fit(myMlpTrainingConfig, trainingDataset, null);

        //then
        doCallRealMethod().when(modelSpy).getModel();
        String actualNumberOfEpochs = modelSpy.getModel().getProperty("Epochs");
        assertThat(actualNumberOfEpochs).isEqualTo(String.valueOf(expectedNumberOfEpochs));
    }

    @Test
    void shouldExecutePredictMethodOnPrediction() throws TranslateException {
        //given
        MyMlpModel myMlpModel = new MyMlpModel(multiLayerPerceptronConfig);
        var modelSpy = spy(myMlpModel);
        SimpleImageDataTranslator translator = mock(SimpleImageDataTranslator.class);
        Image image = mock(Image.class);
        Model model = mock(Model.class);
        doReturn(model).when(modelSpy).getModel();
        var predictor = mock(Predictor.class);
        doReturn(predictor).when(model).newPredictor(any(Translator.class));
        Classifications classifications = mock(Classifications.class);
        doReturn(classifications).when(predictor).predict(any());
        //when
        modelSpy.predict(translator, image);
        //then
        verify(predictor).predict(any());
    }
}