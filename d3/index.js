d3.select("h1")
    .style("color", "blue")
    .attr("class", "heading")
    .text("My New Heading");

d3.select("body")
    .append("p")
    .text("Lorum ipsum blah blah blah.");
