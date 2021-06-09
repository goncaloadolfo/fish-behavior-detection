import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";
import HomePage from "./components/HomePage";
import Login from "./components/Login";
import VideoCard from "./components/VideoCard";
import { BrowserRouter as Router, Route, Switch } from "react-router-dom";

function App() {
  const data = require("./test-data/data.json");
  console.log(data);

  return (
    <div className="App">
      <Router>
        <Switch>
          <Route path="/video-card">
            <VideoCard
              videoDate="08/06/2021 15:37h"
              nSharks={data.nsharks}
              nMantas={data.nmantas}
              sharkDurationHist={data["shark-duration-hist"]}
              mantasDurationHist={data["manta-ray-duration-hist"]}
            ></VideoCard>
          </Route>
          <Route path="/login">
            <Login></Login>
          </Route>
          <Route path="/">
            <HomePage></HomePage>
          </Route>
        </Switch>
      </Router>
    </div>
  );
}

export default App;
