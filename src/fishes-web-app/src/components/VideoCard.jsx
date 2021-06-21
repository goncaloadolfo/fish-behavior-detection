import testVideo from "../test-data/v29.m4v";
import "../css/videoCard.css";
import { Component } from "react";
import { ChevronBarDown, ChevronBarUp } from "react-bootstrap-icons";
import * as d3 from "d3";
import { max, scaleLinear } from "d3";

var moment = require("moment");

const heatmapMargin = { top: 40, right: 0, bottom: 70, left: 0 };
const barchartMargin = { top: 40, right: 20, bottom: 70, left: 60 };

const heatmapWidth = 460 - heatmapMargin.left - heatmapMargin.right;
const heatmapHeight = 380 - heatmapMargin.top - heatmapMargin.bottom;
const barchartWidth = 350 - barchartMargin.left - barchartMargin.right;
const barchartHeight = 350 - barchartMargin.top - barchartMargin.bottom;

class VideoCard extends Component {
  TRANSITION_TIME = 0.3;
  TRANSITION_STEP = 1;

  constructor(props) {
    super(props);
    this.state = {
      visiblePlots: false,
      focusSpecies: "shark",
    };

    this.barchartXAxis = null;
    this.barchartYAxis = null;
    this.barchartSvg = null;

    this.heatmapXAxis = null;
    this.heatmapYAxis = null;
    this.heatmapColorScale = null;
    this.heatmapTooltip = null;
    this.heatmapSvg = null;
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

  getHeatmapMaxValue = () => {
    return max([
      ...this.props.sharkPositionsHist.counts.flat(),
      ...this.props.mantasDurationHist.counts.flat(),
    ]);
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

  setHeatmapXAxis = () => {
    let heatmapXAxis = d3
      .scaleBand()
      .range([0, heatmapWidth])
      .domain(this.props.sharkPositionsHist["xbin_values"])
      .padding(0.05);
    this.heatmapXAxis = heatmapXAxis;

    // this.heatmapSvg
    //   .append("g")
    //   .style("font-size", 15)
    //   .attr("transform", "translate(0," + heatmapHeight + ")")
    //   .call(d3.axisBottom(heatmapXAxis).tickSize(0))
    //   .select(".domain")
    //   .remove();
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

  setupHeatmapYAxis = () => {
    let heatmapYAxis = d3
      .scaleBand()
      .range([heatmapHeight, 0])
      .domain(this.props.sharkPositionsHist["ybin_values"])
      .padding(0.05);
    this.heatmapYAxis = heatmapYAxis;

    // this.heatmapSvg
    //   .append("g")
    //   .style("font-size", 15)
    //   .call(d3.axisLeft(heatmapYAxis).tickSize(0))
    //   .select(".domain")
    //   .remove();
  };

  cleanupBarchartAxis = () => {
    this.barchartSvg.selectAll(".barchart-axis").remove();
  };

  setupHeatmapColorScale = () => {
    this.heatmapColorScale = d3
      .scaleSequential()
      .interpolator(d3.interpolateBlues)
      .domain([0, this.getHeatmapMaxValue()]);
  };

  setupHeatmapTooltip = () => {
    this.heatmapTooltip = d3
      .select("#positions-heatmap")
      .append("div")
      .style("opacity", 0)
      .attr("class", "tooltip")
      .style("background-color", "white")
      .style("border", "solid")
      .style("border-width", "2px")
      .style("border-radius", "5px")
      .style("padding", "5px");
  };

  createBarChart = () => {
    let svg = d3
      .select(`#duration-hist-${this.props.videoId}`)
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
      .attr("class", "h5")
      .attr("text-anchor", "end")
      .attr("x", barchartWidth * 0.8)
      .attr("y", 0)
      .text("Trajectories' positions");

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

  createHeatmap = () => {
    this.heatmapSvg = d3
      .select(`#positions-heatmap-${this.props.videoId}`)
      .append("svg")
      .attr("width", heatmapWidth + heatmapMargin.left + heatmapMargin.right)
      .attr("height", heatmapHeight + heatmapMargin.top + heatmapMargin.bottom)
      .append("g")
      .attr(
        "transform",
        "translate(" + heatmapMargin.left + "," + heatmapMargin.top + ")"
      );

    this.heatmapSvg
      .append("text")
      .attr("class", "h5")
      .attr("text-anchor", "end")
      .attr("x", heatmapWidth * 0.75)
      .attr("y", 0)
      .text("Most frequented regions");

    this.setHeatmapXAxis();
    this.setupHeatmapYAxis();
    this.setupHeatmapColorScale();
    this.setupHeatmapTooltip();
  };

  updatePlotsStatus = () => {
    this.setState({ visiblePlots: !this.state.visiblePlots }, () => {
      let cardElem = document.getElementById(`video-${this.props.videoId}`);
      let currentWidth = parseFloat(cardElem.style.width, 10);
      let step = this.TRANSITION_STEP;

      if (this.state.visiblePlots) {
        let widthDiff = 100 - currentWidth;
        let timeoutTime = (this.TRANSITION_TIME / (widthDiff / step)) * 1000;
        setTimeout(this.increaseCardWidth, 0, cardElem, timeoutTime, step);
      } else {
        let widthDiff = currentWidth - 40;
        let timeoutTime = (this.TRANSITION_TIME / (widthDiff / step)) * 1000;
        setTimeout(this.decreaseCardWidth, 0, cardElem, timeoutTime, step);
      }
    });
  };

  increaseCardWidth = (cardElem, timeoutTime, step) => {
    let currentWidth = parseInt(cardElem.style.width, 10);
    let newWidth = currentWidth + step;
    if (newWidth > 100) newWidth = 100;
    cardElem.style.width = `${newWidth}%`;

    if (newWidth !== 100)
      setTimeout(
        this.increaseCardWidth,
        timeoutTime,
        cardElem,
        timeoutTime,
        step
      );
  };

  decreaseCardWidth = (cardElem, timeoutTime, step) => {
    let currentWidth = parseInt(cardElem.style.width, 10);
    let newWidth = currentWidth - step;
    if (newWidth < 49) newWidth = 49;
    cardElem.style.width = `${newWidth}%`;

    if (newWidth !== 49)
      setTimeout(
        this.decreaseCardWidth,
        timeoutTime,
        cardElem,
        timeoutTime,
        step
      );
  };

  updateFocusSpecies = (e) => {
    this.setState(
      {
        focusSpecies: e.target.innerHTML === "Sharks" ? "shark" : "mantas",
      },
      () => {
        this.updateDurationPlot();
        this.updateHeatmap();
      }
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

  heatmapCellMouseover = function () {
    d3.select(".tooltip").style("opacity", 1);
    d3.select(this).style("stroke", "blue").style("opacity", 1);
  };

  heatmapCellMousemove = function (e) {
    // bug: not working
    d3.select(".tooltip")
      .html(`value: ${e.target.__data__.count}`)
      .style("left", e.clientX + "px")
      .style("bottom", window.innerHeight - e.clientY + "px");
  };

  heatmapCellMouseleave = function () {
    d3.select(".tooltip").style("opacity", 0);
    d3.select(this).style("stroke", "none").style("opacity", 0.8);
  };

  updateHeatmap = () => {
    let heatmapData =
      this.state.focusSpecies === "shark"
        ? this.props.sharkPositionsHist
        : this.props.mantasPositionsHist;

    let data = [];
    for (let i = 0; i < heatmapData.counts.length; i++) {
      for (let j = 0; j < heatmapData.counts[i].length; j++) {
        data.push({
          count: heatmapData.counts[j][i],
          x: heatmapData["xbin_values"][j],
          y: heatmapData["ybin_values"][heatmapData.counts[i].length - i - 1],
        });
      }
    }

    let cellWidth = this.heatmapXAxis.bandwidth();
    let cellHeight = this.heatmapYAxis.bandwidth();

    this.heatmapSvg
      .selectAll()
      .data(data, function (d) {
        return d.x + ":" + d.y;
      })
      .enter()
      .append("rect")
      .attr("x", (d) => this.heatmapXAxis(d.x) + cellWidth / 2)
      .attr("y", (d) => this.heatmapYAxis(d.y) - cellHeight / 2)
      .attr("rx", 4)
      .attr("ry", 4)
      .attr("width", this.heatmapXAxis.bandwidth())
      .attr("height", this.heatmapYAxis.bandwidth())
      .style("fill", (d) => this.heatmapColorScale(d.count))
      .style("stroke-width", 4)
      .style("stroke", "none")
      .style("opacity", 0.8)
      .on("mouseover", this.heatmapCellMouseover)
      .on("mousemove", this.heatmapCellMousemove)
      .on("mouseleave", this.heatmapCellMouseleave);
  };

  dateDisplay = (dateTimeString) => {
    let dateObj = moment(dateTimeString);
    return dateObj.format("LLL");
  };

  timeDiffDisplay = (dateTimeString) => {
    let nowDate = moment(new Date());
    let dateObj = moment(dateTimeString);
    return dateObj.from(nowDate);
  };

  componentDidMount() {
    this.createBarChart();
    this.createHeatmap();
    this.updateDurationPlot();
    this.updateHeatmap();
  }

  render() {
    return (
      <div
        id={`video-${this.props.videoId}`}
        className="card"
        style={{ width: "49%" }}
      >
        <h5 className="card-title">
          {this.dateDisplay(this.props.videoDate)}
          <span className="time-diff">
            {` ${this.timeDiffDisplay(this.props.videoDate)}`}
          </span>
        </h5>
        <h6 className="card-subtitle">
          {this.props.nSharks} Sharks {this.props.nMantas} Mantas
        </h6>
        <video className="video-video-card" src={testVideo} controls></video>

        {this.state.visiblePlots && (
          <ChevronBarUp
            className="expand-icon"
            size={20}
            onClick={this.updatePlotsStatus}
          />
        )}
        {!this.state.visiblePlots && (
          <ChevronBarDown
            className="expand-icon"
            size={20}
            onClick={this.updatePlotsStatus}
          />
        )}

        <div
          id="plot-section"
          style={
            this.state.visiblePlots ? { display: "block" } : { display: "none" }
          }
        >
          <button
            className="sharks-btn species-btn btn btn-outline-secondary"
            style={{
              borderWidth: this.state.focusSpecies === "shark" ? "5px" : "2px",
            }}
            onClick={this.updateFocusSpecies}
          >
            Sharks
          </button>
          <button
            className="species-btn btn btn-outline-secondary"
            style={{
              borderWidth: this.state.focusSpecies === "shark" ? "2px" : "5px",
            }}
            onClick={this.updateFocusSpecies}
          >
            Mantas
          </button>
          <div className="container">
            <div className="row">
              <div className="col">
                <div id={`positions-heatmap-${this.props.videoId}`}></div>
              </div>
              <div className="col">
                <div id={`duration-hist-${this.props.videoId}`}></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }
}

export default VideoCard;
