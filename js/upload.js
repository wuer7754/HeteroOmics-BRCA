const form = document.getElementById('file-form');
const message = document.getElementById('message');

form.addEventListener('submit', (event) => {
  event.preventDefault();

  const file1 = document.getElementById('file1').files[0];
  const file2 = document.getElementById('file2').files[0];
  const file3 = document.getElementById('file3').files[0];

  if (!file1 || !file2 || !file3) {
    message.innerHTML = 'Error: Please select three files.';
    return;
  }

  if (file1.type !== 'text/csv' || file2.type !== 'text/csv' || file3.type !== 'text/csv') {
    message.innerHTML = 'Error: Please select three CSV files.';
    return;
  }

  message.innerHTML = 'Files uploaded successfully!';
});


// Get the elements with the specified IDs
const parameterA = document.getElementById("parameter-a");
const parameterB = document.getElementById("parameter-b");
const parameterC = document.getElementById("parameter-c");

// Check if the sum of the parameters is equal to 1
if (parseFloat(parameterA.value) + parseFloat(parameterB.value) + parseFloat(parameterC.value) !== 1) {
  // Clear the values of the parameters
  parameterA.value = "";
  parameterB.value = "";
  parameterC.value = "";

  // Print an error message
  console.log("Error: The sum of the parameters must be equal to 1.");
}
