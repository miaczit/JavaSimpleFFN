package it.miacz.djl.mnist.example;

import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslatorContext;
import it.miacz.djl.mnist.example.config.MultiLayerPerceptronConfig;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static ai.djl.ndarray.NDManager.newBaseManager;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;

@Tag("unit")
@ExtendWith(MockitoExtension.class)
class SimpleImageDataTranslatorTest {

    @Mock
    MultiLayerPerceptronConfig multiLayerPerceptronConfig;
    @Mock
    TranslatorContext translatorContext;

    @Test
    void shouldCorrectlyProcessInputImage() {
        //given
        SimpleImageDataTranslator simpleImageDataTranslator = new SimpleImageDataTranslator(multiLayerPerceptronConfig);
        try(NDManager ndManager = newBaseManager()){
            NDArray ndArray = ndManager.zeros(new Shape(28, 28, 1));
            var image = mock(Image.class);
            doReturn(ndArray).when(image).toNDArray(any(), any());
            doReturn(ndManager).when(translatorContext).getNDManager();
            //when
            NDList inputList = simpleImageDataTranslator.processInput(translatorContext, image);
            //then
            assertThat(inputList.singletonOrThrow()).isEqualTo(ndArray.transpose());
        }
    }

    @Test
    void shouldReturnExpectedClassification() {
        //given
        SimpleImageDataTranslator simpleImageDataTranslator = new SimpleImageDataTranslator(multiLayerPerceptronConfig);
        TranslatorContext translatorContext = mock(TranslatorContext.class);
        NDList list = mock(NDList.class);
        try(NDManager ndManager = newBaseManager()){
            NDArray ndArray = ndManager.arange(0.0f, 1.0f, 0.1f, DataType.FLOAT64);
            doReturn(ndArray).when(list).singletonOrThrow();
            doReturn(10).when(multiLayerPerceptronConfig).outputSize();
            //when
            Classifications classifications = simpleImageDataTranslator.processOutput(translatorContext, list);
            String actualClassName = classifications.topK().get(0).getClassName();
            //then
            assertThat(actualClassName).isEqualTo("9");
        }

    }

    @Test
    void shouldReturnStackBatchifier() {
        SimpleImageDataTranslator simpleImageDataTranslator = new SimpleImageDataTranslator(multiLayerPerceptronConfig);
        var expectedBatchifier = Batchifier.STACK;

        Batchifier actualBatchifier = simpleImageDataTranslator.getBatchifier();

        assertThat(actualBatchifier).isEqualTo(expectedBatchifier);
    }
}