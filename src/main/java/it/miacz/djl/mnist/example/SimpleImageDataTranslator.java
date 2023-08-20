package it.miacz.djl.mnist.example;

import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import it.miacz.djl.mnist.example.config.MultiLayerPerceptronConfig;
import lombok.RequiredArgsConstructor;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@RequiredArgsConstructor
public class SimpleImageDataTranslator implements Translator<Image, Classifications> {

    private final MultiLayerPerceptronConfig multiLayerPerceptronConfig;

    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.GRAYSCALE);
        return new NDList(NDImageUtils.toTensor(array));
    }

    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list) {
        NDArray probabilities = list.singletonOrThrow().softmax(0);
        List<String> classNames = IntStream.range(0, multiLayerPerceptronConfig.outputSize()).mapToObj(String::valueOf).collect(Collectors.toList());
        return new Classifications(classNames, probabilities);
    }

    @Override
    public Batchifier getBatchifier() {
        return Batchifier.STACK;
    }
}
