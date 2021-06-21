import "../css/videosList.css";
import { Component } from "react";
import VideoCard from "./VideoCard";

class VideosList extends Component {
  constructor(props) {
    super(props);
    let videos = this.fetchVideos();

    this.state = {
      fromDate: undefined,
      toDate: undefined,
      videos: videos,
      filteredVideos: videos,
    };
  }

  updateFromDate = (e) => {
    e.preventDefault();
    let newValue = e.target.value;
    if (newValue !== undefined && newValue !== "")
      this.setState({ fromDate: newValue });
    else this.setState({ fromDate: undefined });
  };

  updateToDate = (e) => {
    e.preventDefault();
    let newValue = e.target.value;
    if (newValue !== undefined && newValue !== "")
      this.setState({ toDate: newValue });
    else this.setState({ toDate: undefined });
  };

  clearDates = (e) => {
    e.preventDefault();
    let dateInputs = document.getElementsByClassName("date-input");
    for (let i = 0; i < dateInputs.length; i++) {
      let dateInputElem = dateInputs.item(i);
      dateInputElem.value = "";
    }

    this.setState({
      fromDate: undefined,
      toDate: undefined,
      filteredVideos: this.state.videos,
    });
  };

  fetchVideos = () => {
    // todo
    let data = require("../test-data/data.json");
    return [data, data, data, data];
  };

  filterVideosByDate = (e) => {
    e.preventDefault();
    if (this.state.fromDate === undefined || this.state.toDate === undefined) {
    }

    let fromDatePassedTime = Date.parse(this.state.fromDate);
    let toDatePassedTime = Date.parse(this.state.toDate);

    let filteredVideos = [];
    this.state.videos.forEach((video) => {
      let videoDatePassedTime = Date.parse("2021-06-16T15:23:05");
      if (
        videoDatePassedTime >= fromDatePassedTime &&
        videoDatePassedTime <= toDatePassedTime
      )
        filteredVideos.push(video);
    });

    this.setState({
      filteredVideos: filteredVideos,
    });
  };

  maxDate = () => {
    let todayDate = new Date();
    let day = todayDate.getDate();
    let month = todayDate.getMonth() + 1;
    let year = todayDate.getFullYear();

    if (day < 10) day = "0" + day;
    if (month < 10) month = "0" + month;
    return year + "-" + month + "-" + day;
  };

  isSearchButtonEnabled = () => {
    return this.state.fromDate !== undefined && this.state.toDate !== undefined;
  };

  searchButtonCursor = () => {
    return this.state.fromDate !== undefined && this.state.toDate !== undefined
      ? "initial"
      : "not-allowed";
  };

  render() {
    return (
      <div className="tab-section">
        <div id="video-list-header" style={{ backgroundColor: "white" }}>
          <h4 className="tab-title">Videos</h4>
          <table
            id="inputs-table"
            className="table table-sm table-borderless d-flex"
            style={{ margin: 0 }}
          >
            <tbody>
              <tr>
                <td>
                  <label className="fw-bolder" htmlFor="fromDate">
                    from
                  </label>
                </td>
                <td>
                  <input
                    className="date-input"
                    type="date"
                    name="fromDate"
                    max={this.maxDate()}
                    onChange={this.updateFromDate}
                  ></input>
                </td>
              </tr>

              <tr>
                <td>
                  <label className="fw-bolder" htmlFor="fromDate">
                    to
                  </label>
                </td>
                <td>
                  <input
                    className="date-input"
                    type="date"
                    name="fromDate"
                    max={this.maxDate()}
                    onChange={this.updateToDate}
                  ></input>
                </td>
              </tr>
            </tbody>
          </table>

          <div className="search-clear-btns">
            <button
              id="search-btn"
              className="btn btn-outline-secondary fs-6 search-clear-btn"
              onClick={this.filterVideosByDate}
              disabled={!this.isSearchButtonEnabled()}
            >
              search
            </button>
            <button
              id="clear-btn"
              className="btn btn-outline-secondary fs-6 search-clear-btn"
              onClick={this.clearDates}
            >
              clear
            </button>
          </div>
        </div>

        <div className="container-flex">
          <div className="row" style={{ width: "100%", margin: 0 }}>
            {this.state.filteredVideos.map((video, index) => {
              return (
                <VideoCard
                  key={`video-${index}`}
                  videoId={index}
                  videoDate="2021-06-16T15:23:05"
                  nSharks={video.nsharks}
                  nMantas={video.nmantas}
                  sharkDurationHist={video["shark-duration-hist"]}
                  mantasDurationHist={video["manta-ray-duration-hist"]}
                  sharkPositionsHist={video["shark-positions-hist"]}
                  mantasPositionsHist={video["manta-ray-positions-hist"]}
                ></VideoCard>
              );
            })}
          </div>
        </div>
      </div>
    );
  }
}

export default VideosList;
