import * as tf from "@tensorflow/tfjs";

document.querySelector(".inputs");

tf.ready().then(() => {
  const modelPath = "./simple-ttt-model/model/ttt_model.json";
  tf.tidy(() => {
    tf.loadLayersModel(modelPath).then((model) => {
      // Three board states
      const emptyBoard = tf.zeros([9]);
      const betterBlockMe = tf.tensor([-1, 0, 0, 1, 1, -1, 0, 0, -1]);
      const goForTheKill = tf.tensor([1, 0, 1, 0, -1, -1, -1, 0, 1]);

      // Stack states into a shape [3, 9]
      const matches = tf.stack([emptyBoard, betterBlockMe, goForTheKill]);
      const result = model.predict(matches);
      // Log the results
      result.reshape([3, 3, 3]).print();
    });
  });
});

// Tensor[

//En este caso con el tablero vacio se puede ver que el modelo predice que el juego distribuido
//en todo el tablero

//   ([
//     [0.2287459, 0.0000143, 0.2659601],
//     [0.0000982, 0.0041204, 0.0001773],
//     [0.2301052, 0.0000206, 0.270758],
//   ],

// En este caso con el tablero de mejor bloqueo se puede ver que el modelo predice que el proximo movimiento sera en la esquina superior derecha
//
//   [
//     [0.0011957, 0.0032045, 0.9908957],
//     [0.000263, 0.0006491, 0.0000799],
//     [0.0010194, 0.0002893, 0.0024035],
//   ],

// En este caso con el tablero de ganar para el otro jugador se puede ver que el modelo predice que el proximo movimiento sera en la posicion del centro de la primera fila
//   [
//     [0.0000056, 0.9867876, 0.0000028],
//     [0.0003809, 0.0001524, 0.0011258],
//     [0.0000328, 0.0114983, 0.0000139],
//   ])
// ];
