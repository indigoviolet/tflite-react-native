
package com.reactlibrary;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;


import com.facebook.react.bridge.Callback;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

import com.indigoviolet.react.ArrayUtil;

public class TfliteReactNativeModule extends ReactContextBaseJavaModule {

  private final ReactApplicationContext reactContext;
  private Interpreter tfLite;
  private int inputSize = 0;
  float[][] labelProb;
  private static final int BYTES_PER_CHANNEL = 4;


  public TfliteReactNativeModule(ReactApplicationContext reactContext) {
    super(reactContext);
    this.reactContext = reactContext;
  }

  @Override
  public String getName() {
    return "TfliteReactNative";
  }

  @ReactMethod
  private void loadModel(final String modelPath, final int numThreads, final Callback callback)
      throws IOException {
    AssetManager assetManager = reactContext.getAssets();
    AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    MappedByteBuffer buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

    final Interpreter.Options tfliteOptions = new Interpreter.Options();
    tfliteOptions.setNumThreads(numThreads);
    tfLite = new Interpreter(buffer, tfliteOptions);

    callback.invoke(null, "success");
  }

  private ByteBuffer feedInputTensorImage(String path, int rotation, float mean, float std) throws IOException {
    Tensor tensor = tfLite.getInputTensor(0);
    inputSize = tensor.shape()[1];
    int inputChannels = tensor.shape()[3];

    InputStream inputStream = new FileInputStream(path.replace("file://", ""));
    Bitmap bitmapRaw = BitmapFactory.decodeStream(inputStream);

    Matrix matrix = getTransformationMatrix(bitmapRaw.getWidth(), bitmapRaw.getHeight(),
                                            inputSize, inputSize, false, rotation);

    int[] intValues = new int[inputSize * inputSize];
    int bytePerChannel = tensor.dataType() == DataType.UINT8 ? 1 : BYTES_PER_CHANNEL;
    ByteBuffer imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * inputChannels * bytePerChannel);
    imgData.order(ByteOrder.nativeOrder());

    Bitmap bitmap = Bitmap.createBitmap(bitmapRaw, 0, 0, bitmapRaw.getWidth(), bitmapRaw.getHeight(), matrix, true);
    bitmap.setConfig(Bitmap.Config.ARGB_8888);
    // final Canvas canvas = new Canvas(bitmap);
    // canvas.drawBitmap(bitmapRaw, matrix, null);
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    // TODO - this is posenet specific
    int pixel = 0;
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        int pixelValue = intValues[pixel++];
        if (tensor.dataType() == DataType.FLOAT32) {
          imgData.putFloat((((pixelValue >> 16) & 0xFF) - mean) / std);
          imgData.putFloat((((pixelValue >> 8) & 0xFF) - mean) / std);
          imgData.putFloat(((pixelValue & 0xFF) - mean) / std);
        } else {
          imgData.put((byte) ((pixelValue >> 16) & 0xFF));
          imgData.put((byte) ((pixelValue >> 8) & 0xFF));
          imgData.put((byte) (pixelValue & 0xFF));
        }
      }
    }

    return imgData;
  }

  @ReactMethod
  private void runModelOnImageMulti(final String path, final int rotation, final float mean,
                                    final float std, final Callback callback) throws IOException {

    ByteBuffer imgData = feedInputTensorImage(path, rotation, mean, std);
    Object[] input = new Object[]{imgData};
    Map<Integer, Object> outputMap = makeOutputMap(float.class);
    tfLite.runForMultipleInputsOutputs(input, outputMap);

    callback.invoke(null, ArrayUtil.toWritableArray(outputMap.values().toArray()));
  }

  private Map<Integer, Object> makeOutputMap(Class<?> componentType) {
    Map<Integer, Object> outputMap = new HashMap<>();
    for (int i = 0; i < tfLite.getOutputTensorCount(); i++) {
      int[] shape = tfLite.getOutputTensor(i).shape();
      Object output = Array.newInstance(componentType, shape);
      outputMap.put(i, output);
    }
    return outputMap;
  }

  @ReactMethod
  private void close() {
    tfLite.close();
    labelProb = null;
  }

  private static Matrix getTransformationMatrix(final int srcWidth,
                                                final int srcHeight,
                                                final int dstWidth,
                                                final int dstHeight,
                                                final boolean maintainAspectRatio,
                                                final int rotation
                                                ) {
    final Matrix matrix = new Matrix();

    if (srcWidth != dstWidth || srcHeight != dstHeight) {
      final float scaleFactorX = dstWidth / (float) srcWidth;
      final float scaleFactorY = dstHeight / (float) srcHeight;

      if (maintainAspectRatio) {
        final float scaleFactor = Math.max(scaleFactorX, scaleFactorY);
        matrix.postScale(scaleFactor, scaleFactor);
      } else {
        matrix.postScale(scaleFactorX, scaleFactorY);
      }
    }

    matrix.invert(new Matrix());
    if (rotation != 0) {
      matrix.postRotate(rotation);
    }

    return matrix;
  }

}
