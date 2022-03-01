<?php

namespace core\form;

use core\Model;

class Field{
    private const TEXT_FIELD = 'text';
    private const EMAIL_FIELD = 'email';
    private const PASSWORD_FIELD = 'password';
    private const DATE_FIELD = 'date';
    private const NUMBER_FIELD = 'number';
    private const PHONE_FIELD = 'tel';

    public Model $model;
    public string $attribute;

    private string $type;

    public function __construct(Model $model, string $attribute,){
        $this->model = $model;
        $this->attribute = $attribute;
        $this->type = self::TEXT_FIELD;
        
    }

    protected function picker(){
        if ($this->type === self::NUMBER_FIELD){
            return "min='{$this->min}' max='{$this->max}'";
        }

        if ($this->type === self::PHONE_FIELD){
            return (isset($this->pattern) && !empty($this->patten)) ? "pattern = '{$this->pattern}'" : '';
        }
    }

    public function __toString(){
        return \sprintf('
            <div class="input-group flex-wrap mt-1">
                <span class="input-group-text">%s</span>
                <input type="%s" name="%s" value="%s" class="form-control %s" placeholder="%s" %s/>
                <div class="invalid-feedback font-monospace">
                    %s
                </div>
            </div>
        ',
            $this->attribute,
            $this->type,
            $this->attribute,
            $this->model->{$this->attribute},
            $this->model->hasError($this->attribute) ? 'is-invalid' : '',
            $this->attribute,
            $this->picker(),
            $this->model->getFirstError($this->attribute)
        );
    }

    public function passwordField(){
        $this->type = self::PASSWORD_FIELD;
        return $this;
    }

    public function emailField(){
        $this->type = self::EMAIL_FIELD;
        return $this;
    }


    public function phoneField(string $pattern){
        $this->type = self::PHONE_FIELD;
        $this->$pattern = $pattern;
        return $this;
    }

    public function dateField(){
        $this->type = self::DATE_FIELD;
        return $this;
    }
    public function numberField($min, $max){
        $this->type = self::NUMBER_FIELD;
        $this->min = $min;
        $this->max = $max;
        return $this;
    }
}