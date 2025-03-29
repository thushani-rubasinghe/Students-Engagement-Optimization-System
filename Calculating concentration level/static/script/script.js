//alert("You are redirecting to the Increase Video Resolution Page")
//
//let processing_window = document.getElementById('processing_window')
//let output = document.getElementById('status');
//let progress_bar = document.getElementById('progress_bar')
//processing_window.style.visibility = "hidden"
//
//function sleep(ms) {
//      return new Promise(resolve => setTimeout(resolve, ms));
//   }
//
//
//async function startProcess() {
//    processing_window.style.visibility = "visible"
//
//    for (let i = 1; i < 1200 ; i++) {
//         await sleep(1000);
//         fetch('http://127.0.0.1:5000/start_process').then((response) => response.json())
//            .then((responseData) => {
//              console.log(responseData.result);
//              output.innerHTML = "Status: " + responseData.result;
//              progress_bar.innerHTML = responseData.progress + "%";
//              progress_bar.style.width = responseData.progress + "%"
//            })
//            .catch(error => console.warn(error));
//          }
//                 console.log(data);
//}