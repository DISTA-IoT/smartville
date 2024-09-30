// Function to show the content of a specific section based on the provided sectionId
function showContent(sectionId) {
    console.log("Attempting to display the section with ID:", sectionId); // Log the ID of the section being displayed

    // Get all elements with class 'content'
    var contents = document.getElementsByClassName('content');
    
    // Loop through all content sections and hide them
    for (var i = 0; i < contents.length; i++) {
        contents[i].style.display = 'none'; // Hide each content section
    }

    // Get the selected section by its ID
    var selectedSection = document.getElementById(sectionId);
    
    // Check if the section exists, if yes, display it
    if (selectedSection) {
        selectedSection.style.display = 'block'; // Show the selected section
        console.log("Section found:", selectedSection); // Log success message
    } else {
        console.error("Section with ID '" + sectionId + "' not found."); // Log error if section does not exist
    }
}

// Function to toggle the visibility of the Code subcategories
function toggleSubcategories() {
    var codeSubcategories = document.getElementById('codeSubcategories');
    
    // Check if the 'codeSubcategories' div is currently visible
    if (codeSubcategories.style.display === 'block') {
        codeSubcategories.style.display = 'none'; // Hide subcategories if already visible
    } else {
        codeSubcategories.style.display = 'block'; // Show subcategories if hidden
    }
}

// Function to toggle the visibility of the pox_controller subcategories
function togglePoxControllerSubcategories() {
    var poxControllerSubcategories = document.getElementById('poxControllerSubcategories');
    
    // Check if the 'poxControllerSubcategories' div is currently visible
    if (poxControllerSubcategories.style.display === 'block') {
        poxControllerSubcategories.style.display = 'none'; // Hide subcategories if already visible
    } else {
        poxControllerSubcategories.style.display = 'block'; // Show subcategories if hidden
    }
}

// Event listener to handle mouse wheel scrolling inside the sidebar
document.getElementById('sidebar').addEventListener('wheel', function(event) {
    // Normalize the scroll delta to ensure consistent behavior across different browsers
    const delta = Math.sign(event.deltaY);

    // Adjust the scroll amount (increase or decrease the value for smoother or faster scrolling)
    const scrollAmount = 30;

    // Scroll the sidebar based on the scroll delta
    this.scrollBy(0, delta * scrollAmount);

    // Prevent the default scrolling behavior of the entire page
    event.preventDefault();
});
