import testVideo from "../test-data/v29.m4v";
import { Component } from "react";
import { ChevronBarDown, ChevronBarUp } from "react-bootstrap-icons";
import * as d3 from "d3";
import { max, scaleLinear } from "d3";

const plotMargin = { top: 30, right: 30, bottom: 70, left: 60 };
const plotWidth = 480 - plotMargin.left - plotMargin.right;
const plotHeight = 400 - plotMargin.top - plotMargin.bottom;

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
      .range([0, plotWidth])
      .domain(this.getBarchartLabels())
      .padding(0.2);
    this.barchartSvg
      .append("g")
      .attr("class", "barchart-axis")
      .attr("transform", `translate(0, ${plotHeight})`)
      .call(d3.axisBottom(xAxis));
    this.barchartXAxis = xAxis;
  };

  setHeatmapXAxis = () => {
    let heatmapXAxis = d3
      .scaleBand()
      .range([0, plotWidth])
      .domain(this.props.sharkPositionsHist["xbin_values"])
      .padding(0.05);
    this.heatmapXAxis = heatmapXAxis;

    this.heatmapSvg
      .append("g")
      .style("font-size", 15)
      .attr("transform", "translate(0," + plotHeight + ")")
      .call(d3.axisBottom(heatmapXAxis).tickSize(0))
      .select(".domain")
      .remove();
  };

  setBarchartYAxis = () => {
    let yAxis = scaleLinear()
      .domain([0, this.getBarchartMaxValue() + 5])
      .range([plotHeight, 0]);
    this.barchartSvg
      .append("g")
      .attr("class", "barchart-axis")
      .call(d3.axisLeft(yAxis));
    this.barchartYAxis = yAxis;
  };

  setupHeatmapYAxis = () => {
    let heatmapYAxis = d3
      .scaleBand()
      .range([plotHeight, 0])
      .domain(this.props.sharkPositionsHist["ybin_values"])
      .padding(0.05);
    this.heatmapYAxis = heatmapYAxis;

    this.heatmapSvg
      .append("g")
      .style("font-size", 15)
      .call(d3.axisLeft(heatmapYAxis).tickSize(0))
      .select(".domain")
      .remove();
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
      .select("#duration-hist")
      .append("svg")
      .attr("width", plotWidth + plotMargin.left + plotMargin.right)
      .attr("height", plotHeight + plotMargin.top + plotMargin.bottom)
      .append("g")
      .attr("transform", `translate(${plotMargin.left}, ${plotMargin.top})`);

    this.barchartSvg = svg;
    this.setBarchartXAxis();
    this.setBarchartYAxis();

    svg
      .append("text")
      .attr("class", "title")
      .attr("text-anchor", "end")
      .attr("x", plotWidth / 2)
      .attr("y", 0)
      .text("Trajectories Duration");

    svg
      .append("text")
      .attr("class", "x-label")
      .attr("text-anchor", "end")
      .attr("x", plotWidth)
      .attr("y", plotHeight + 45)
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
      .select("#positions-heatmap")
      .append("svg")
      .attr("width", plotWidth + plotMargin.left + plotMargin.right)
      .attr("height", plotHeight + plotMargin.top + plotMargin.bottom)
      .append("g")
      .attr(
        "transform",
        "translate(" + plotMargin.left + "," + plotMargin.top + ")"
      );
    this.setHeatmapXAxis();
    this.setupHeatmapYAxis();
    this.setupHeatmapColorScale();
    this.setupHeatmapTooltip();
  };

  updatePlotsStatus = () => {
    this.setState({ visiblePlots: !this.state.visiblePlots });
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
      .attr("height", (d) => plotHeight - this.barchartYAxis(d.count))
      .attr("fill", "#69b3a2")
      .attr("transform", `translate(${this.barchartXAxis.bandwidth() / 2}, 0)`);
  };

  heatmapCellMouseover = function () {
    d3.select(".tooltip").style("opacity", 1);
    d3.select(this).style("stroke", "blue").style("opacity", 1);
  };

  heatmapCellMousemove = function (e) {
    d3.select(".tooltip")
      .html(`value: ${e.target.__data__.count}`)
      .style("left", e.x + "px")
      .style("bottom", e.y + "px");
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
          y: heatmapData["ybin_values"][i],
        });
      }
    }

    this.heatmapSvg
      .selectAll()
      .data(data, function (d) {
        return d.x + ":" + d.y;
      })
      .enter()
      .append("rect")
      .attr("x", (d) => this.heatmapXAxis(d.x))
      .attr("y", (d) => this.heatmapYAxis(d.y))
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

  componentDidMount() {
    this.createBarChart();
    this.createHeatmap();
    this.updateDurationPlot();
    this.updateHeatmap();
  }

  render() {
    return (
      <div className="card" style={{ width: "50%" }}>
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
          <div id="positions-heatmap"></div>
        </div>
      </div>
    );
  }
}

export default VideoCard;
