package main

import (
	service "github.com/nayotta/metathings-sensor-led-display/pkg/led-display/service"
	component "github.com/nayotta/metathings/pkg/component"
)

func main() {
	mdl, err := component.NewModule("led-display", new(service.LEDDisplayService))
	if err != nil {
		panic(err)
	}

	err = mdl.Launch()
	if err != nil {
		panic(err)
	}
}


