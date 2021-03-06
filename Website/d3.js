//map modified from https://bl.ocks.org/wboykinm/dbbe50d1023f90d4e241712395c27fb3
//Width and height of map
var width = 900;
var height = 600;

var lowColor = '#8CD2E8'
var highColor = '#1E3F66'

// D3 Projection
var projection = d3.geoAlbersUsa()
  .translate([width / 2, height / 2]) // translate to center of screen
  .scale([1000]); // scale things down so see entire US

// Define path generator
var path = d3.geoPath() // path generator that will convert GeoJSON to SVG paths
  .projection(projection); // tell path generator to use albersUsa projection

//Create SVG element and append map to the SVG
var svg = d3.select("#map")
  .append("svg")
  .attr("width", width)
  .attr("height", height);

//Add tooltip variable div
// Append Div for tooltip to SVG
var div = d3.select("#map")
		    .append("div")   
    		.attr("class", "tooltip")               
    		.style("opacity", 0);

// Load in data
d3.csv("stateCount.csv", function(data) {
	var dataArray = [];
	for (var d = 0; d < data.length; d++) {
		dataArray.push(parseFloat(data[d].value))
	}
	var minVal = d3.min(dataArray)
	var maxVal = d3.max(dataArray)
	var ramp = d3.scaleLinear().domain([minVal,maxVal]).range([lowColor,highColor])
	
  // Load GeoJSON data and merge with states data
  d3.json("us-states.json", function(json) {

    // Loop through each state data value in the .csv file
    for (var i = 0; i < data.length; i++) {

      // Grab State Name
      var dataState = data[i].state;

      // Grab data value 
      var dataValue = data[i].value;

      // Grab data value 
      var dataDivision = data[i].division;

      
      // Find the corresponding state inside the GeoJSON
      for (var j = 0; j < json.features.length; j++) {
        var jsonState = json.features[j].properties.name;

        if (dataState == jsonState) {

          // Copy the data value into the JSON
          json.features[j].properties.value = dataValue;

          // Copy the division name into the JSON
          json.features[j].properties.division = dataDivision;

          // Stop looking through the JSON
          break;
        }
      }
    }

    // Bind the data to the SVG and create one path per GeoJSON feature
    svg.selectAll("path")
      .data(json.features)
      .enter()
      .append("path")
      .attr("d", path)
      // Modification of custom tooltip code provided by Malcolm Maclean, "D3 Tips and Tricks" 
      // http://www.d3noob.org/2013/01/adding-tooltips-to-d3js-graph.html
      .on("mouseover", function(d) {      
          div.transition()        
              .duration(200)      
              .style("opacity", .9);      
          var tooltip_text = d.properties.name + "<br>" + d.properties.value + " KWH"    
          div.html(tooltip_text)
              .style("left", (d3.event.pageX) + "px")     
              .style("top", (d3.event.pageY - 20) + "px");   
      })   

        // fade out tooltip on mouse out               
        .on("mouseout", function(d) {       
            div.transition()        
              .duration(500)      
              .style("opacity", 0);   
        })
      .style("stroke", "#fff")
      .style("stroke-width", "1")
      .style("fill", function(d) { return ramp(d.properties.value)})
    });
  });
