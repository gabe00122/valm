import Button from "@mui/material/Button";
import "./App.css";
import { useEffect, useState } from "react";

function App() {
    const [data, setData] = useState({});

    useEffect(() => {
        async function loadEpisode() {
            const response = await fetch("/api/episode/0");
            const json = await response.json();
            setData(json);
        }

        loadEpisode();
    }, []);

    return (
        <div>
            <Button variant="contained">Hello world</Button>
            <pre>{JSON.stringify(data, null, 2)}</pre>
        </div>
    );
}

export default App;
