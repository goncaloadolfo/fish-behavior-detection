import { Component } from "react";
import {
  CameraReelsFill,
  ExclamationLg,
  InfoLg,
  ConeStriped,
} from "react-bootstrap-icons";
import VideosList from "./VideosList";
import logo from "../images/oceanarium-logo.png";
import "../css/videosPage.css";

class VideosListPage extends Component {
  render() {
    return (
      <div className="container-fluid" style={{ margin: 0, padding: 0 }}>
        <div className="row" style={{ width: "100%", margin: 0 }}>
          <div
            className="col-2"
            style={{ backgroundColor: "white", padding: 0 }}
          >
            <div className="text-center">
              <img src={logo} width="40%" alt="oceanarium logo"></img>
            </div>
            <ul className="nav flex-column main-nav">
              <li className="nav-item main-nav-item">
                <CameraReelsFill className="nav-icon" size={20} />
                <span className="fw-bolder fs-6 align-middle">Videos</span>
              </li>
              <li className="nav-item main-nav-item">
                <ExclamationLg className="nav-icon" size={20} />
                <span className="fw-bolder fs-6 align-middle">Alerts</span>
              </li>
              <li className="nav-item main-nav-item">
                <InfoLg className="nav-icon" size={20} />
                <span className="fw-bolder fs-6 align-middle">
                  Interesting Moments
                </span>
              </li>
              <li className="nav-item main-nav-item">
                <ConeStriped className="nav-icon" size={20} />
                <span className="fw-bolder fs-6 align-middle">
                  Surface Warnings
                </span>
              </li>
            </ul>
          </div>
          <div className="col-10" style={{ padding: 0 }}>
            <VideosList></VideosList>
          </div>
        </div>
      </div>
    );
  }
}

export default VideosListPage;
