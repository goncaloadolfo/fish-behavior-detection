import testVideo from "../test-data/v29.m4v";
import { Component } from "react";
import { ChevronBarDown, ChevronBarUp } from "react-bootstrap-icons";
import * as d3 from "d3";
import { max, scaleLinear } from "d3";

const barchartMargin = { top: 30, right: 30, bottom: 70, left: 60 };
const barchartWidth = 480 - barchartMargin.left - barchartMargin.right;
const barchartHeight = 400 - barchartMargin.top - barchartMargin.bottom;

class VideoCard extends Component {
  constructor(props) {
    super(props);
    this.state = {
      visiblePlots: false,
      focusSpecies: "shark",
    };
    this.barchartXAxis = null;
    this.barchartYAxis = null;
    this.barchartSvg = null;
  }

  getBarchartLabels = () => {
    let speciesHistogram =
      this.state.focusSpecies === "shark"
        ? this.props.sharkDurationHist
        : this.props.mantasDurationHist;
    return speciesHistogram["bin-values"].map((x) => x.toFixed(0));
  };

  getBarchartMaxValue = () => {
    let speciesHistogram =
      this.state.focusSpecies === "shark"
        ? this.props.sharkDurationHist
        : this.props.mantasDurationHist;
    return max(speciesHistogram.counts);
  };

  setBarchartXAxis = () => {
    let xAxis = d3
      .scaleBand()
      .range([0, barchartWidth])
      .domain(this.getBarchartLabels())
      .padding(0.2);
    this.barchartSvg
      .append("g")
      .attr("class", "barchart-axis")
      .attr("transform", `translate(0, ${barchartHeight})`)
      .call(d3.axisBottom(xAxis));
    this.barchartXAxis = xAxis;
  };

  setBarchartYAxis = () => {
    let yAxis = scaleLinear()
      .domain([0, this.getBarchartMaxValue() + 5])
      .range([barchartHeight, 0]);
    this.barchartSvg
      .append("g")
      .attr("class", "barchart-axis")
      .call(d3.axisLeft(yAxis));
    this.barchartYAxis = yAxis;
  };

  cleanupBarchartAxis = () => {
    this.barchartSvg.selectAll(".barchart-axis").remove();
  };

  createBarChart = () => {
    let svg = d3
      .select("#duration-hist")
      .append("svg")
      .attr("width", barchartWidth + barchartMargin.left + barchartMargin.right)
      .attr(
        "height",
        barchartHeight + barchartMargin.top + barchartMargin.bottom
      )
      .append("g")
      .attr(
        "transform",
        `translate(${barchartMargin.left}, ${barchartMargin.top})`
      );

    this.barchartSvg = svg;
    this.setBarchartXAxis();
    this.setBarchartYAxis();

    svg
      .append("text")
      .attr("class", "title")
      .attr("text-anchor", "end")
      .attr("x", barchartWidth / 2)
      .attr("y", 0)
      .text("Trajectories Duration");

    svg
      .append("text")
      .attr("class", "x-label")
      .attr("text-anchor", "end")
      .attr("x", barchartWidth)
      .attr("y", barchartHeight + 45)
      .text("trajectory time (s)");

    svg
      .append("text")
      .attr("class", "y label")
      .attr("text-anchor", "end")
      .attr("y", -45)
      .attr("dy", ".75em")
      .attr("transform", "rotate(-90)")
      .text("number of trajectories");
  };

  updatePlotsStatus = () => {
    this.setState({ visiblePlots: !this.state.visiblePlots });
    console.log(this.state.visiblePlots);
  };

  updateFocusSpecies = (e) => {
    this.setState(
      {
        focusSpecies: e.target.innerHTML === "Sharks" ? "shark" : "mantas",
      },
      () => this.updateDurationPlot()
    );
  };

  updateDurationPlot = () => {
    this.cleanupBarchartAxis();
    this.setBarchartXAxis();
    this.setBarchartYAxis();

    let speciesHistogram =
      this.state.focusSpecies === "shark"
        ? this.props.sharkDurationHist
        : this.props.mantasDurationHist;
    let data = speciesHistogram.counts.map((value, index) => {
      return {
        count: value,
        label: speciesHistogram["bin-values"][index].toFixed(0),
      };
    });

    let aux = this.barchartSvg.selectAll("rect").data(data);
    aux
      .join("rect")
      .transition()
      .duration(1000)
      .attr("x", (d) => this.barchartXAxis(d.label))
      .attr("y", (d) => this.barchartYAxis(d.count))
      .attr("width", this.barchartXAxis.bandwidth())
      .attr("height", (d) => barchartHeight - this.barchartYAxis(d.count))
      .attr("fill", "#69b3a2")
      .attr("transform", `translate(${this.barchartXAxis.bandwidth() / 2}, 0)`);
  };

  componentDidMount() {
    this.createBarChart();
    this.updateDurationPlot();
  }

  render() {
    return (
      <div className="card">
        <h5 className="card-title">{this.props.videoDate}</h5>
        <h6 className="card-subtitle">
          {this.props.nSharks} Sharks {this.props.nMantas} Mantas
        </h6>
        <video src={testVideo} controls></video>

        {this.state.visiblePlots && (
          <ChevronBarUp onClick={this.updatePlotsStatus} />
        )}
        {!this.state.visiblePlots && (
          <ChevronBarDown onClick={this.updatePlotsStatus} />
        )}

        <div
          id="plot-section"
          style={
            this.state.visiblePlots ? { display: "block" } : { display: "none" }
          }
        >
          <button onClick={this.updateFocusSpecies}>Sharks</button>
          <button onClick={this.updateFocusSpecies}>Mantas</button>
          <div id="duration-hist"></div>
          <div id="positions-2dhist"></div>
        </div>
      </div>
    );
  }
}

export default VideoCard;
