import { NativeModules, Image } from 'react-native';

const { TfliteReactNative } = NativeModules;

class Tflite {
  loadModel(args, callback) {
    TfliteReactNative.loadModel(args['model'], args['numThreads'] || 1, (error, response) => {
      callback && callback(error, response);
    });
  }

  runModelOnImageMulti(args, callback) {
    TfliteReactNative.runModelOnImageMulti(
      args['path'],
      args['imageMean'] != null ? args['imageMean'] : 127.5,
      args['imageStd'] != null ? args['imageStd'] : 127.5,
      (error, response) => {
        callback && callback(error, response);
      }
    );
  }

  close() {
    TfliteReactNative.close();
  }
}

export default Tflite;
