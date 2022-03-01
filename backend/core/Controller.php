<?php
namespace core;

abstract class Controller{
    private ?string $Layout = Null;
    private array $Middlewares = [];
    protected Request $request;
    protected Response $response;

    public function setMiddleWare(MiddleWare $middleware){
        $this->Middlewares[] = $middleware;
    }

    public function middlewares(): array
    {
        return $this->Middlewares;
    }

    public function layout(): ?string
    {
        return $this->Layout;
    }

    public function setLayout($layout){
        $this->Layout = $layout;
    }

    public function render($view, $params = []): array|bool|string|null
    {
        return Application::APP()->router->renderView($view, $params);
    }
}